#!/usr/bin/env python3
"""
FCL Semantic Search Prototype BETA - Family Law

A semantic search system for UK Family Division case law with Citizens Advice tagging

Requirements:
    - XML files in ./caselaw_fam_xml/
    - ca_terms_all.pkl in current directory
    - AWS credentials configured (AWS SSO) with access to the S3 Vectors bucket
"""

import sys
import os
import re
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import boto3
from botocore.exceptions import ClientError
import streamlit as st

# ============================================================================
# CONFIGURATION
# ============================================================================

XML_DIR = "./caselaw_xml"
VECTORS_PATH = "./caselaw_vectors_2026.pkl"
CA_TERMS_PATH = "./beta-app-folder/ca_terms_all.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# S3 Vectors configuration
AWS_REGION = "eu-west-2"
S3_VECTOR_BUCKET_NAME = "caselaw-semantic-shift-beta"
S3_VECTOR_INDEX_NAME = "caselaw-semantic-beta-all" 

# S3 Vectors query_vectors currently caps top_k at 30 per call, with no pagination.
S3_VECTORS_MAX_TOPK = 30

# Akoma Ntoso XML namespaces
NS = {
    "akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0",
    "uk": "https://caselaw.nationalarchives.gov.uk/akn"
}

# ============================================================================
# XML PARSING
# ============================================================================

def _get_text(parent, tag):
    """Extract text from an XML element."""
    el = parent.find(tag, NS)
    return el.text.strip() if el is not None and el.text else None


def parse_caselaw_xml(filepath: Path) -> Tuple[Optional[Dict], Optional[List[str]]]:
    """
    Parse a Find Case Law XML file (Akoma Ntoso format).
    
    Returns:
        Tuple of (metadata_dict, list_of_paragraphs)
    """
    try:
        root = ET.parse(filepath).getroot()
    except ET.ParseError as e:
        print(f"  Warning: Could not parse {filepath.name}: {e}")
        return None, None

    # Extract metadata
    meta = {}
    
    frbrwork = root.find(".//akn:FRBRWork", NS)
    if frbrwork is not None:
        date_el = frbrwork.find("akn:FRBRdate", NS)
        meta["date"] = date_el.get("date") if date_el is not None else None
        name_el = frbrwork.find("akn:FRBRname", NS)
        meta["case_name"] = name_el.get("value") if name_el is not None else None
        if meta.get("date"):
            try:
                meta["year"] = int(meta["date"].split("-")[0])
            except (ValueError, IndexError):
                meta["year"] = None

    prop = root.find(".//akn:proprietary", NS)
    if prop is not None:
        meta["court"] = _get_text(prop, "uk:court")
        meta["neutral_citation"] = _get_text(prop, "uk:cite")
        meta["case_number"] = _get_text(prop, "uk:caseNumber")

    judge_el = root.find(".//akn:judge", NS)
    meta["judge"] = judge_el.text.strip() if judge_el is not None and judge_el.text else None

    # Extract paragraphs
    paragraphs = []
    body = root.find(".//akn:judgmentBody", NS)
    if body is not None:
        for para in body.findall(".//akn:paragraph", NS):
            text = " ".join(para.itertext()).strip()
            if text and len(text) > 40:
                paragraphs.append(text)

    return meta, paragraphs


# ============================================================================
# CASE-LEVEL TAGGING
# ============================================================================

def tag_judgment(judgment_embedding: np.ndarray, ca_terms: List[Dict], threshold: float = 0.40) -> List[str]:
    """
    Tag an entire judgment based on similarity to CA term definitions.
    
    Args:
        judgment_embedding: numpy array (384,) - embedding of judgment sample
        ca_terms: list of CA term dicts with 'embedding' key
        threshold: minimum cosine similarity (default 0.40)
    
    Returns:
        List of tag names that match
    """
    if not ca_terms:
        return []
    
    tags = []
    judgment_emb_2d = judgment_embedding.reshape(1, -1)
    
    for term_data in ca_terms:
        term_emb = np.array(term_data['embedding']).reshape(1, -1)
        similarity = cosine_similarity(judgment_emb_2d, term_emb)[0][0]
        
        if similarity >= threshold:
            tags.append({
                'term': term_data['term'],
                'similarity': similarity
            })
    
    # Sort by similarity (highest first)
    tags.sort(key=lambda x: x['similarity'], reverse=True)

    # Keep only top 3 tags
    top_tags = tags[:4]

    return [tag['term'] for tag in top_tags]


# ============================================================================
# EMBEDDING COMMAND
# ============================================================================

def cmd_embed():
    """Embed all XML files and tag with CA terms."""
    print("="*80)
    print("STEP 1: EMBEDDING CASE LAW WITH CA TAGGING")
    print("="*80)
    
    # Load CA terms
    try:
        with open(CA_TERMS_PATH, 'rb') as f:
            ca_terms = pickle.load(f)
        print(f"\n✓ Loaded {len(ca_terms)} Citizens Advice terms for tagging")
    except FileNotFoundError:
        print(f"\n⚠️  CA terms file not found at {CA_TERMS_PATH}")
        print("    Continuing without tagging.")
        ca_terms = []
    
    # Find XML files
    xml_files = list(Path(XML_DIR).glob("*.xml"))
    print(f"✓ Found {len(xml_files)} XML file(s) in {XML_DIR}")
    
    if not xml_files:
        print(f"\n❌ No .xml files found in {XML_DIR}")
        print("   Make sure your XML files are in the correct directory.")
        sys.exit(1)
    
    # Load embedding model
    print(f"\n⏳ Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✓ Model ready\n")
    
    # Process files
    all_vectors = []
    tag_stats = {
        'total_cases': 0,
        'tagged_cases': 0,
        'total_tags': 0,
        'total_paragraphs': 0
    }
    
    for i, filepath in enumerate(xml_files, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(xml_files)} cases...")
        
        meta, paragraphs = parse_caselaw_xml(filepath)
        
        if meta is None or not paragraphs:
            continue
        
        doc_id = filepath.stem
        tag_stats['total_cases'] += 1
        
        # CASE-LEVEL TAGGING
        # Use first 20 paragraphs or 3000 words as representative sample
        sample_paras = paragraphs[:20]
        sample_text = " ".join(sample_paras)
        words = sample_text.split()[:3000]
        judgment_sample = " ".join(words)
        
        # Embed WITHOUT context for tagging
        judgment_embedding = model.encode(judgment_sample, show_progress_bar=False)
        
        # Tag the judgment
        case_tag_names = tag_judgment(judgment_embedding, ca_terms, threshold=0.35)
        
        if case_tag_names:
            tag_stats['tagged_cases'] += 1
            tag_stats['total_tags'] += len(case_tag_names)
        
        # PARAGRAPH-LEVEL EMBEDDING (with context for search)
        for j, para_text in enumerate(paragraphs):
            tag_stats['total_paragraphs'] += 1
            
            # Prepend metadata context for better search
            context_prefix = (
                f"Court: {meta.get('court', 'Unknown')}. "
                f"Judge: {meta.get('judge', 'Unknown')}. "
                f"Date: {meta.get('date', 'Unknown')}. "
                f"Case: {meta.get('case_name', 'Unknown')}.\n\n"
            )
            
            para_embedding = model.encode(
                context_prefix + para_text,
                show_progress_bar=False
            )
            
            all_vectors.append({
                "id": f"{doc_id}_para_{j}",
                "values": para_embedding.tolist(),
                "metadata": {
                    "court": meta.get("court") or "",
                    "judge": meta.get("judge") or "",
                    "date": meta.get("date") or "",
                    "year": meta.get("year") or 0,
                    "case_name": meta.get("case_name") or "",
                    "neutral_citation": meta.get("neutral_citation") or "",
                    "case_number": meta.get("case_number") or "",
                    "doc_id": doc_id,
                    "chunk_index": j,
                    "text": para_text[:1500],
                    "tags": case_tag_names,
                }
            })
    
    # Save vectors
    print(f"\n⏳ Saving {len(all_vectors)} vectors to {VECTORS_PATH}...")
    with open(VECTORS_PATH, "wb") as f:
        pickle.dump(all_vectors, f)
    
    print(f"✓ Saved {len(all_vectors)} vectors")
    
    # Print statistics
    print(f"\n{'='*80}")
    print("📊 EMBEDDING STATISTICS:")
    print("="*80)
    print(f"  Total cases processed: {tag_stats['total_cases']}")
    print(f"  Total paragraphs: {tag_stats['total_paragraphs']:,}")
    print(f"  Total vectors created: {len(all_vectors):,}")
    
    if ca_terms:
        print(f"\n📊 TAGGING STATISTICS (Case-Level):")
        print(f"  Cases tagged: {tag_stats['tagged_cases']}/{tag_stats['total_cases']} "
              f"({100*tag_stats['tagged_cases']/tag_stats['total_cases']:.1f}%)")
        print(f"  Total tags applied: {tag_stats['total_tags']}")
        if tag_stats['tagged_cases'] > 0:
            print(f"  Average tags per case: {tag_stats['total_tags']/tag_stats['tagged_cases']:.1f}")
    
    print(f"\n✅ Embedding complete! Vectors saved to {VECTORS_PATH}")
    print("="*80)


# ============================================================================
# UPSERT COMMAND
# ============================================================================

def cmd_upsert():
    """Upload vectors to S3 Vectors. Resumable: if interrupted (e.g. SSO
    session expiry on a long run), re-running this command picks up from the
    last completed batch rather than starting over."""
    print("="*80)
    print("STEP 2: UPLOADING TO S3 VECTORS")
    print("="*80)

    # Load vectors
    print(f"\n⏳ Loading vectors from {VECTORS_PATH}...")
    try:
        with open(VECTORS_PATH, "rb") as f:
            all_vectors = pickle.load(f)
        print(f"✓ Loaded {len(all_vectors):,} vectors")
    except FileNotFoundError:
        print(f"\n❌ Vectors file not found: {VECTORS_PATH}")
        print("   Run 'python beta_app.py embed' first")
        sys.exit(1)

    # Checkpoint file — one per vectors file + index combo, so switching
    # corpora or indexes doesn't collide with an unrelated checkpoint.
    checkpoint_path = Path(f"{VECTORS_PATH}.{S3_VECTOR_INDEX_NAME}.upsert_checkpoint")
    start_batch = 0
    if checkpoint_path.exists():
        try:
            start_batch = int(checkpoint_path.read_text().strip())
            print(f"\n↻ Found checkpoint — resuming from batch {start_batch}")
        except ValueError:
            print(f"\n⚠️  Could not read checkpoint file, starting from the beginning")
            start_batch = 0

    # Connect to S3 Vectors (uses AWS SSO / default credential chain — no keys in code)
    print(f"\n⏳ Connecting to S3 Vectors bucket '{S3_VECTOR_BUCKET_NAME}', "
          f"index '{S3_VECTOR_INDEX_NAME}'...")
    s3vectors = boto3.client("s3vectors", region_name=AWS_REGION)
    print("✓ Connected")

    TEXT_BYTE_LIMIT = 1400  # issue encountered during upsert, leaves ~650 bytes headroom for all other fields combined 

    def safe_truncate_bytes(s: str, max_bytes: int) -> str:
        encoded = s.encode("utf-8")
        if len(encoded) <= max_bytes:
            return s
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    def to_s3_vector(v):
        metadata = dict(v["metadata"])  # shallow copy so we don't mutate the original
        if not metadata.get("tags"):
            metadata.pop("tags", None)
        if "text" in metadata and isinstance(metadata["text"], str):
            metadata["text"] = safe_truncate_bytes(metadata["text"], TEXT_BYTE_LIMIT)
        return {
            "key": v["id"],
            "data": {"float32": v["values"]},
            "metadata": metadata,
        }

    # S3 Vectors put_vectors accepts up to 500 vectors per call
    batch_size = 500
    total_batches = (len(all_vectors) + batch_size - 1) // batch_size

    # Log of any individual records that had to be skipped (e.g. still too
    # large after truncation, or some other per-record validation issue) so
    # nothing silently vanishes from the corpus without a record of it.
    skip_log_path = Path(f"{VECTORS_PATH}.{S3_VECTOR_INDEX_NAME}.skipped_vectors.txt")
    skipped_count = 0

    print(f"\n⏳ Upserting {len(all_vectors):,} vectors in {total_batches} batches "
          f"(starting at batch {start_batch})...")

    batch_num = 0
    try:
        for i in range(0, len(all_vectors), batch_size):
            if batch_num < start_batch:
                batch_num += 1
                continue  # already uploaded in a previous run

            batch = [to_s3_vector(v) for v in all_vectors[i:i + batch_size]]

            # Attempt the batch; if AWS rejects one specific record (e.g. a
            # validation error naming a single key), remove just that record
            # and retry the rest of the batch, rather than losing the whole
            # batch or stopping the run. Capped at a sane number of retries
            # per batch as a safety net against an unexpected repeating error.
            for attempt in range(20):
                if not batch:
                    break  # everything in this batch got skipped
                try:
                    s3vectors.put_vectors(
                        vectorBucketName=S3_VECTOR_BUCKET_NAME,
                        indexName=S3_VECTOR_INDEX_NAME,
                        vectors=batch,
                    )
                    break  # success
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    error_message = e.response.get("Error", {}).get("Message", "")
                    match = re.search(r"Invalid record for key '([^']+)'", error_message)
                    if error_code == "ValidationException" and match:
                        bad_key = match.group(1)
                        batch = [b for b in batch if b["key"] != bad_key]
                        skipped_count += 1
                        with open(skip_log_path, "a") as logf:
                            logf.write(f"{bad_key}\t{error_message}\n")
                        continue  # retry the batch without the bad record
                    raise  # some other error — let the outer handler deal with it
            else:
                raise RuntimeError(
                    f"Batch {batch_num} still failing after 20 skip attempts — "
                    f"stopping rather than looping indefinitely."
                )

            batch_num += 1
            # Save progress after every successful batch, so an interruption
            # at any point can resume from here rather than the start.
            checkpoint_path.write_text(str(batch_num))

            if batch_num % 10 == 0:
                print(f"  Progress: {min(i + batch_size, len(all_vectors))}/{len(all_vectors)} "
                      f"vectors uploaded (batch {batch_num}/{total_batches}, "
                      f"{skipped_count} skipped so far)...")

    except Exception as e:
        print(f"\n❌ Upload interrupted at batch {batch_num}/{total_batches}: {e}")
        print(f"   Progress saved. Re-run 'python beta_app.py upsert' to resume "
              f"from batch {batch_num}.")
        print(f"   (If this was an SSO token error, run 'aws sso login --profile "
              f"<your-profile>' first.)")
        sys.exit(1)

    # All batches completed — remove the checkpoint so a future fresh run
    # (e.g. re-embedding and re-uploading later) starts clean.
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n✅ Upload complete! {len(all_vectors) - skipped_count:,} vectors now in S3 Vectors")
    if skipped_count:
        print(f"   ⚠️  {skipped_count} record(s) were skipped (individually rejected by "
              f"S3 Vectors even after truncation). See {skip_log_path} for details.")
    print("="*80)


# ============================================================================
# SEARCH COMMAND (STREAMLIT UI)
# ============================================================================

def cmd_search():
    """Launch Streamlit search interface."""
    print("="*80)
    print("STEP 3: LAUNCHING SEARCH INTERFACE")
    print("="*80)
    print("\n⏳ Starting Streamlit app...")
    print("   The search interface will open in your browser automatically.")
    print("   Press Ctrl+C to stop the server.\n")
    
    # Run Streamlit
    os.system(f"streamlit run {__file__} --server.headless=true")


# ============================================================================
# STREAMLIT UI (runs when called via streamlit run)
# ============================================================================

def build_pinecone_filter(court: str, judge: str, year_from: int, year_to: int, citation: str) -> Optional[Dict]:
    """Build Pinecone metadata filter."""
    conditions = []
    
    if court.strip():
        conditions.append({"court": {"$eq": court.strip()}})
    if judge.strip():
        conditions.append({"judge": {"$eq": judge.strip().upper()}})
    if year_from:
        conditions.append({"year": {"$gte": year_from}})
    if year_to:
        conditions.append({"year": {"$lte": year_to}})
    if citation.strip():
        conditions.append({"neutral_citation": {"$eq": citation.strip()}})
    
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def render_ca_tags(tag_names: List[str], ca_lookup: Dict) -> str:
    """Render CA term tags as HTML with tooltips."""
    if not tag_names:
        return ""
    
    tag_html_parts = []
    for tag_name in tag_names[:6]:  # Limit to 6 tags
        if tag_name in ca_lookup:
            # Tag with definition and link
            definition = ca_lookup[tag_name]['definition']
            source = ca_lookup[tag_name]['source']
            
            # Escape quotes
            definition_escaped = definition.replace('"', '&quot;').replace("'", "&#39;")
            
            # Truncate long definitions
            if len(definition_escaped) > 300:
                definition_escaped = definition_escaped[:300] + "..."
            
            tag_html_parts.append(f"""<span style="display: inline-block; background: #e8f4f8; color: #2c5f7a; padding: 3px 8px; margin-right: 4px; margin-bottom: 4px; border-radius: 10px; font-size: 0.80em; cursor: help; border: 1px solid #d0e8f0;" title="{definition_escaped}"><a href="{source}" target="_blank" style="color: inherit; text-decoration: none;">{tag_name} ℹ️</a></span>""")
        else:
            # Tag without definition (just display the name)
            tag_html_parts.append(f"""<span style="display: inline-block; background: #f0f0f0; color: #666; padding: 3px 8px; margin-right: 4px; margin-bottom: 4px; border-radius: 10px; font-size: 0.80em; border: 1px solid #ddd;">{tag_name}</span>""")
    
    if len(tag_names) > 6:
        tag_html_parts.append(f"""<span style="font-size:0.80em; color:#888; font-style:italic; margin-left: 4px;">+{len(tag_names) - 6} more</span>""")
    
    # Wrap in flex container for horizontal layout
    return f'<div style="display: flex; flex-wrap: wrap; align-items: center;">{"".join(tag_html_parts)}</div>'


def streamlit_ui():
    """Main Streamlit search interface."""
    st.set_page_config(page_title="Lost for Words BETA: case law case search", page_icon="⚖️", layout="wide")
    
    st.title("⚖️ Lost for Words BETA: family law case search")
    st.markdown("Semantic search prototype for Find Case Law with plain English legal concept tags from Citizens Advice")
    
    # Load resources
    @st.cache_resource
    def load_model():
        # Webapp mode must never fetch the model over the network — it's
        # baked into the image at build time. Fail fast if it's missing
        # rather than silently falling back to a download attempt.
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

    @st.cache_resource
    def load_ca_terms():
        try:
            with open(CA_TERMS_PATH, 'rb') as f:
                terms = pickle.load(f)
            lookup = {}
            for term in terms:
                lookup[term['term']] = {
                    'definition': term['definition'],
                    'source': term['source']
                }
            return lookup
        except FileNotFoundError:
            return {}
    
    @st.cache_resource
    def connect_s3vectors():
        return boto3.client("s3vectors", region_name=AWS_REGION)
    
    model = load_model()
    ca_lookup = load_ca_terms()
    s3vectors = connect_s3vectors()
    
    # Search input
    st.markdown("---")
    
    query = st.text_input("🔍 Search", placeholder="e.g., parental responsibility dispute", label_visibility="collapsed")

    # Set fixed number of results to fetch.
    top_k = 20  # number of CASES to display 
    query_top_k = S3_VECTORS_MAX_TOPK  # number of raw paragraph matches to fetch (max 30)

    # Search on enter or button click
    if query:
        with st.spinner("Searching..."):
            # Encode query
            query_vector = model.encode(query, show_progress_bar=False).tolist()

            # Search S3 Vectors (get as many paragraphs as allowed, to aggregate by case)
            response = s3vectors.query_vectors(
                vectorBucketName=S3_VECTOR_BUCKET_NAME,
                indexName=S3_VECTOR_INDEX_NAME,
                queryVector={"float32": query_vector},
                topK=query_top_k,
                returnMetadata=True,
                returnDistance=True,
            )

            # Translate S3 Vectors response shape into the {score, metadata} shape
            # the rest of the app expects. For cosine distance, similarity = 1 - distance
            # (S3 Vectors returns distance; Pinecone returned similarity score directly).
            raw_matches = []
            for v in response.get("vectors", []):
                distance = v.get("distance", 0.0)
                similarity = 1 - distance
                raw_matches.append({
                    "score": similarity,
                    "metadata": v.get("metadata", {}),
                })

            # Aggregate scores by case (Option 3)
            from collections import defaultdict
            case_scores = defaultdict(list)

            for match in raw_matches:
                case_id = match["metadata"].get("doc_id")
                case_scores[case_id].append({
                    'score': match['score'],
                    'match': match
                })
            
            # Rank cases by average score
            ranked_cases = []
            for case_id, paragraphs in case_scores.items():
                avg_score = sum(p['score'] for p in paragraphs) / len(paragraphs)
                max_score = max(p['score'] for p in paragraphs)
                best_para = max(paragraphs, key=lambda x: x['score'])['match']
                
                ranked_cases.append({
                    'case_id': case_id,
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'num_matches': len(paragraphs),
                    'best_paragraph': best_para
                })
            
            # Sort by average score and take top N
            ranked_cases.sort(key=lambda x: x['avg_score'], reverse=True)
            matches = [c['best_paragraph'] for c in ranked_cases[:top_k]]
            
            # Display results
            if not matches:
                st.info("No results found. Try a different query.")
            else:
                st.success(f"Found {len(matches)} unique cases") 
                # Initialize pagination state
                if 'results_shown' not in st.session_state:
                    st.session_state.results_shown = 10
    
                # Display results from 0 to results_shown
                results_to_show = matches[:st.session_state.results_shown]
    
                # ... rest of result display code ...
                
                # Show tagging info
                tagged_count = sum(1 for m in matches if m['metadata'].get('tags'))
                if ca_lookup and tagged_count > 0:
                    st.info(f"💡 {tagged_count}/{len(matches)} results have legal concept tags. Click tags to read more on Citizens Advice.")
                
                st.markdown("---")
                
                # Render each result
                for i, match in enumerate(matches, 1):
                    m = match["metadata"]
                    score = match["score"]
                    
                    # Find this case's info from ranked_cases
                    case_info = next((c for c in ranked_cases if c['best_paragraph'] == match), None)
                    num_matching_paras = case_info['num_matches'] if case_info else 1
                    avg_score = case_info['avg_score'] if case_info else score
                    
                    text = m.get("text", "")
                    preview = text[:600] + ("..." if len(text) > 600 else "")
                        
                    # Case name and score
                    col_left, col_right = st.columns([3, 1])
                    
                    # Get link to full judgment
                    if m.get("doc_id"):
                        path = m["doc_id"].replace("_", "/")
                        fcl_url = f"https://caselaw.nationalarchives.gov.uk/{path}"
                        case_link = f"[{m.get('case_name', 'Unknown case')}]({fcl_url})"
                    else:
                        case_link = m.get('case_name', 'Unknown case')
                    
                    with col_left:
                        st.markdown(f"### {i}. {case_link}")
                        if num_matching_paras > 1:
                            st.caption(f"{num_matching_paras} matching paragraphs in this case")
                    with col_right:
                        st.markdown(f"**{avg_score:.3f}** avg", help=f"Average similarity across {num_matching_paras} matching paragraph(s). Individual paragraph scores range from {min(p['score'] for p in case_scores[m.get('doc_id')]):.3f} to {max(p['score'] for p in case_scores[m.get('doc_id')]):.3f}. Scores: 1.0 = perfect match, 0.0 = no match. Scores above 0.7 are generally very relevant.")
                        
                    # Tags
                    tag_names = m.get("tags", [])
                    if tag_names and ca_lookup:
                        tags_html = render_ca_tags(tag_names, ca_lookup)
                        st.markdown(tags_html, unsafe_allow_html=True)
                        
                    # Metadata
                    meta_parts = []
                    if m.get("court"): meta_parts.append(f"**Court:** {m['court']}")
                    if m.get("judge"): meta_parts.append(f"**Judge:** {m['judge']}")
                    if m.get("date"): meta_parts.append(f"**Date:** {m['date']}")
                    if m.get("neutral_citation"): meta_parts.append(f"**Citation:** {m['neutral_citation']}")
                        
                    if meta_parts:
                        st.markdown(" · ".join(meta_parts))
                    
                    # Paragraph text (excerpt) - plain text display
                    with st.expander("📄 Read excerpt", expanded=True):
                        st.write(preview)
                    
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""""
                **About:** Semantic search powered by sentence transformers. Legal concept tags from [Citizens Advice](https://www.citizensadvice.org.uk/family/). 
                This is a prototype semantic search engine that uses vector embeddings to match search queries to case law judgments. 
                Vectors were created at the paragraph level, the paragraph with theclosest semantic match to the search query is shown in the preview. 
                Clicking on a  suggested case takes you to the judgement on the [Find Case Law](https://caselaw.nationalarchives.gov.uk/) website.
                This tool is intended for research and prototyping purposes only.
                This is an alpha version developed by Caitlin Wilson as part of a collaborative PhD project between King's College London and The National Archives, funded by the London Arts and Humanities Partnership.
                Supervisors: Dr Barbara McGillivray and Dr Niccolo Ridi.
                With generous support from the Find Case Law team at The National Archives.
                """"") 

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main command-line interface."""
    
    # Check for CLI commands FIRST (before checking for Streamlit)
    if len(sys.argv) < 2:
        # No command provided - check if running via streamlit
        if "streamlit" in sys.modules:
            streamlit_ui()
            return
        else:
            print(__doc__)
            sys.exit(1)
    
    # Handle CLI commands
    command = sys.argv[1].lower()
    
    # If it's a Streamlit command, run Streamlit
    if command == "search" and "streamlit" not in sys.argv[0]:
        # User said 'search' - launch streamlit
        cmd_search()
        return
    
    # Otherwise handle CLI commands
    if command == "embed":
        cmd_embed()
    elif command == "upsert":
        cmd_upsert()
    elif command == "search":
        cmd_search()
    elif command == "all":
        cmd_embed()
        cmd_upsert()
        cmd_search()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()