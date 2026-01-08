# RAG System Improvement Plan
## Issue: Not Getting Answers to Questions

### Critical Issues Found

1. **LLM Prompt Too Restrictive** ⚠️ HIGH PRIORITY
   - Current prompt tells LLM to say "unable to answer" even when information exists
   - Overly strict grounding rules prevent the model from answering valid questions
   - Need to balance accuracy with helpfulness

2. **Retrieval May Be Too Strict**
   - MIN_RELEVANCE_SCORE = 0.2 might still filter out relevant chunks
   - TOP_K_RETRIEVAL = 15 may not be enough for complex queries
   - Adaptive threshold of 50% of best score might be too high

3. **Keyword Matching Limitations**
   - No stemming (e.g., "report" vs "reports")
   - Limited synonym handling
   - May miss partial matches

4. **Model Name Issue**
   - GPT-5.2 may not exist or may have different API name
   - Need to verify correct model identifier

---

## Action Plan

### Phase 1: Critical Fixes (Do First)

#### 1. Fix LLM Prompt (HIGHEST PRIORITY)
**Problem**: Prompt is too restrictive, causing LLM to refuse valid answers
**Solution**: 
- Modify prompt to be more balanced
- Allow LLM to answer when information is clearly present
- Only say "unable" when truly no relevant info exists
- Add instruction to try harder to find answers in context

**File**: `backend/app/core/rag/generator.py` - `_build_messages()` method

#### 2. Verify Model Name
**Problem**: GPT-5.2 may not be valid model name
**Solution**:
- Test if `gpt-5.2` works
- If not, try: `gpt-4o`, `gpt-4-turbo`, `gpt-4o-2024-08-06`
- Update config with correct model name

**File**: `backend/app/core/config.py`

#### 3. Increase Retrieval Coverage
**Problem**: May not be retrieving enough chunks
**Solution**:
- Increase `TOP_K_RETRIEVAL` from 15 to 20
- Lower `MIN_RELEVANCE_SCORE` from 0.2 to 0.15
- Increase fallback from top 10 to top 15 chunks
- Lower adaptive threshold from 50% to 40% of best score

**Files**: 
- `backend/app/core/config.py`
- `backend/app/api/routes/chat.py`

---

### Phase 2: Enhanced Retrieval (Do Next)

#### 4. Improve Keyword Matching
**Problem**: Missing matches due to word variations
**Solution**:
- Add stemming (use NLTK or simple stemmer)
- Expand synonym dictionary with technical terms
- Support partial word matches (e.g., "certif" matches "certificate")
- Extract 4-word phrases (currently only 3-word)
- Add acronym expansion (HACCP, GMP, etc.)

**File**: `backend/app/core/rag/retriever.py` - `_keyword_search()` method

#### 5. Enhance Query Expansion
**Problem**: Query variations may not cover all terminology
**Solution**:
- Add more technical synonyms
- Industry-specific term mappings
- Common abbreviation expansions
- Related concept suggestions

**File**: `backend/app/core/rag/query_understanding.py`

#### 6. Improve Vector Search
**Problem**: Embeddings may not capture semantic similarity well
**Solution**:
- Consider upgrading embedding model (e.g., `all-mpnet-base-v2` for better quality)
- Increase candidate multiplier from 2.5x to 3x
- Add reranking step using cross-encoder model
- Try different similarity metrics

**Files**:
- `backend/app/core/config.py` - EMBEDDING_MODEL
- `backend/app/core/rag/retriever.py` - `_vector_search()`

---

### Phase 3: Debugging & Monitoring (Ongoing)

#### 7. Add Comprehensive Logging
**Problem**: Hard to debug why answers aren't found
**Solution**:
- Log all retrieved chunks with scores
- Log why chunks were filtered out
- Log LLM prompt (truncated)
- Log final context sent to LLM
- Log LLM response before processing

**Files**: 
- `backend/app/api/routes/chat.py`
- `backend/app/core/rag/generator.py`
- `backend/app/core/rag/retriever.py`

#### 8. Add Fallback Strategies
**Problem**: If initial search fails, no recovery
**Solution**:
- If no chunks found, try broader search (remove filters)
- If still nothing, search all documents
- If still nothing, return top chunks regardless of score
- Provide helpful error with suggestions

**File**: `backend/app/api/routes/chat.py`

#### 9. Create Test Suite
**Problem**: No way to verify improvements
**Solution**:
- Create test cases with known questions and expected chunks
- Test retrieval quality
- Test end-to-end query processing
- Track success rate over time

**File**: `backend/tests/test_retrieval.py` (new)

---

### Phase 4: Advanced Improvements (Future)

#### 10. Add Reranking
- Use cross-encoder model to rerank top candidates
- Better semantic matching than simple score combination

#### 11. Query Understanding
- Parse question type (what, how, when, where, why)
- Extract entities and relationships
- Generate multiple query formulations

#### 12. Context Window Optimization
- Dynamically adjust chunk size based on query complexity
- Prioritize high-relevance chunks in context
- Truncate low-relevance chunks more aggressively

---

## Implementation Priority

### Immediate (Do Today):
1. ✅ Fix LLM prompt - make it less restrictive
2. ✅ Verify/fix model name
3. ✅ Increase retrieval coverage (TOP_K, thresholds)

### Short-term (This Week):
4. Improve keyword matching
5. Add comprehensive logging
6. Add fallback strategies

### Medium-term (Next Week):
7. Enhance query expansion
8. Improve vector search
9. Create test suite

### Long-term (Future):
10. Add reranking
11. Advanced query understanding
12. Context optimization

---

## Success Metrics

Track these metrics to measure improvement:
- **Answer Rate**: % of queries that get answers (target: >80%)
- **Retrieval Quality**: % of retrieved chunks that are relevant (target: >70%)
- **User Satisfaction**: Positive feedback ratio (target: >75%)
- **Response Time**: Average latency (target: <5s)

---

## Testing Checklist

After each change, test with:
- [ ] Simple factual question (e.g., "What is the temperature requirement?")
- [ ] Complex multi-part question
- [ ] Question with typos
- [ ] Question using different terminology
- [ ] Question about specific document section
- [ ] Question that should return "unable to answer" (negative test)

---

## Notes

- Always test changes incrementally
- Monitor logs after each change
- Keep backup of working configuration
- Document any model-specific behavior

