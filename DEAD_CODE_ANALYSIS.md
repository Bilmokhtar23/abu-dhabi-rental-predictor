# Dead Code Analysis - Abu Dhabi Rental Predictor

**Analysis Date:** November 25, 2025
**Purpose:** Identify unused/dead code before showing project to employers

---

## SUMMARY

**Total Python Files:** 30
**Active Files:** 11 (37%)
**Dead/Unused Files:** 19 (63%)
**Recommendation:** Remove or clearly archive dead code

---

## CATEGORY 1: ACTUALLY USED (Keep) ✅

### Core Application (3 files)
1. **app.py** - Main Streamlit application ✅ ACTIVE
2. **scripts/train_stacked_ensemble.py** - Training pipeline ✅ ACTIVE  
3. **scripts/audit_pipeline.py** - Pipeline validation ✅ MIGHT BE USED

### Tests (8 files) 
4. **tests/conftest.py** - Test fixtures ✅ ACTIVE
5. **tests/test_app.py** - App tests ✅ ACTIVE
6. **tests/test_data/test_loader.py** - Data tests ✅ ACTIVE
7. **tests/test_features/test_feature_engineering.py** - Feature tests ✅ ACTIVE
8. **tests/test_models/test_predictor.py** - Model tests ✅ ACTIVE
9. **tests/test_scripts/test_training.py** - Training tests ✅ ACTIVE
10. **tests/test_src/test_utils.py** - Utils tests ✅ ACTIVE (imports src/utils)
11. **tests/test_monitoring/test_monitoring.py** - Monitoring placeholders ⚠️ SKIPPED

---

## CATEGORY 2: DEAD CODE (Remove or Archive) ❌

### Feature Engineering - UNUSED (3 files)
12. **src/features/feature_engineering.py** (278 lines)
   - Contains sophisticated FeatureEngineer class
   - Generates 38 features
   - Never imported anywhere
   - **VERDICT:** DELETE - This is from an earlier iteration

13. **src/features/feature_engineering_v2.py** (similar)
   - Another version of feature engineering
   - Never imported
   - **VERDICT:** DELETE - Also from earlier iteration

14. **src/features/__init__.py**
   - Empty or minimal
   - **VERDICT:** CAN DELETE if above files are removed

### Models - BROKEN (2 files)
15. **src/models/predictor.py** (contains RentalPredictor class)
   - References non-existent v6.2 model paths
   - Imports break if called
   - Never used by app.py or tests
   - **VERDICT:** DELETE - Leftover from earlier architecture

16. **src/models/__init__.py**
   - **VERDICT:** CAN DELETE if predictor.py is removed

### API - INCOMPLETE (2 files)
17. **src/api/main.py** (FastAPI skeleton)
   - Has POST /predict endpoint defined
   - Has rate limiting and schemas
   - Never integrated with app.py
   - **VERDICT:** Either IMPLEMENT properly or DELETE

18. **src/api/schemas.py** (Pydantic models)
   - Defines PredictionRequest/Response
   - Only used by main.py (which isn't used)
   - **VERDICT:** DELETE (unless implementing API)

### Data Loaders - UNUSED (3 files)
19. **src/data/data_loader.py**
   - Functions to load train/val/test splits
   - Never imported (app.py and training script load data directly)
   - **VERDICT:** DELETE or REFACTOR into shared utility

20. **src/data/validators.py**
   - Pydantic validators for property data
   - Never used
   - **VERDICT:** DELETE

21. **src/data/__init__.py**
   - **VERDICT:** CAN DELETE if above removed

### Utils - MINIMAL USE (2 files)
22. **src/utils/feature_engineering.py**
   - Only imported by tests/test_src/test_utils.py
   - Functions: create_log_features, create_interaction_features
   - **VERDICT:** KEEP (tested) BUT note it's only used by tests

23. **src/utils/__init__.py**
   - **VERDICT:** KEEP (if keeping utils/)

### Config - PARTIALLY USED (2 files)
24. **config/__init__.py**
   - May contain get_api_config, get_webapp_config
   - Only imported by src/api/main.py (which isn't used)
   - **VERDICT:** CHECK if used elsewhere, likely can simplify

25. **config/model_config.py**
   - Model paths configuration
   - **VERDICT:** CHECK if used by training script

### Monitoring - PLACEHOLDER (1 file)
26. **src/monitoring/__init__.py**
   - Appears to be placeholder
   - Tests are skipped
   - **VERDICT:** Either IMPLEMENT or DELETE

### Scripts - ONE UNUSED (2 files)
27. **scripts/apply_final_feature_set.py**
   - Appears to be a one-time data processing script
   - **VERDICT:** CHECK if still needed, likely archive

28. **scripts/__init__.py**
   - **VERDICT:** KEEP

### Package Init Files (2 files)
29. **src/__init__.py** - DELETE if removing src/
30. **tests/__init__.py** - KEEP

---

## DETAILED RECOMMENDATIONS

### HIGH PRIORITY: DELETE THESE (Clean Repo)

```bash
# Feature engineering from earlier iterations (NEVER USED)
rm src/features/feature_engineering.py
rm src/features/feature_engineering_v2.py
rm src/features/__init__.py

# Broken model predictor (REFERENCES NON-EXISTENT PATHS)
rm src/models/predictor.py  
rm src/models/__init__.py

# Data loaders (NOT USED - app loads data directly)
rm src/data/data_loader.py
rm src/data/validators.py
rm src/data/__init__.py

# Monitoring placeholder (NOT IMPLEMENTED)
rm src/monitoring/__init__.py
```

**Result:** Removes 9 files, ~500-800 lines of dead code

---

### MEDIUM PRIORITY: DECIDE ON THESE

#### Option A: DELETE API (If not planning to use)
```bash
rm src/api/main.py
rm src/api/schemas.py
rm -rf src/api/
```

#### Option B: IMPLEMENT API (If planning to use)
- Integrate src/api/main.py with actual model loading
- Add to requirements.txt as optional
- Document in README as "API mode"
- Test it works

**Recommendation:** DELETE for now. Can add back later if needed.

---

### LOW PRIORITY: CLEANUP

1. **scripts/apply_final_feature_set.py**
   - Move to `archive/` directory if keeping for reference
   - Delete if no longer relevant

2. **config/__init__.py** and **config/model_config.py**
   - Verify what training script actually uses
   - Simplify if not needed

3. **src/utils/feature_engineering.py**
   - Only used by tests
   - Consider if worth keeping

---

## AFTER CLEANUP: PROJECT STRUCTURE

```
Property finder/
├── app.py                          # Main Streamlit app
├── requirements.txt
├── config/
│   └── model_config.py             # If actually used
├── scripts/
│   └── train_stacked_ensemble.py   # Training pipeline
├── tests/
│   ├── conftest.py
│   ├── test_app.py
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   └── test_scripts/
├── data/
│   └── processed/
└── model_outputs/
    └── production/
```

**Result:** Much cleaner, easier to navigate, professional

---

## WHY THIS MATTERS FOR HIRING

### Current State (With Dead Code):
- ❌ Suggests incomplete cleanup
- ❌ Hard to navigate for reviewers  
- ❌ Unclear what's active vs archived
- ❌ Raises questions: "Why is this here?"

### After Cleanup:
- ✅ Shows attention to detail
- ✅ Clear project structure
- ✅ Easy for reviewers to understand
- ✅ Professional polish

---

## IMPLEMENTATION PLAN

### Phase 1: Safe Deletions (30 minutes)
1. Create backup branch: `git checkout -b cleanup-dead-code`
2. Delete clearly unused files (feature_engineering.py, predictor.py, etc.)
3. Run tests: `pytest tests/ -v` (should still pass)
4. Test app: `streamlit run app.py` (should still work)

### Phase 2: Verify & Commit (15 minutes)
5. Review git diff
6. Commit: "Remove dead code from earlier iterations"
7. Merge to main if tests pass

### Phase 3: Documentation Update (15 minutes)
8. Update README if project structure changed
9. Add CHANGELOG.md entry if tracking changes

**Total Time: 1 hour**

---

## VALIDATION CHECKLIST

After cleanup, verify:
- [ ] All tests pass: `pytest tests/ -v`
- [ ] App launches: `streamlit run app.py`
- [ ] Training works: `python scripts/train_stacked_ensemble.py`
- [ ] No import errors in any active files
- [ ] README reflects actual project structure

---

## QUESTIONS TO ANSWER BEFORE DELETING

1. **Is src/api/main.py planned for future use?**
   - If YES: Keep but add TODO comment
   - If NO: Delete

2. **Is config/ actually used by training script?**
   - Run: `grep -n "from config" scripts/train_stacked_ensemble.py`
   - If not found: Can simplify/delete

3. **Should we keep src/utils/ for future?**
   - Currently only used by tests
   - Probably keep for now

---

## FINAL RECOMMENDATION

**DELETE 9-12 files immediately:**
- All src/features/ (unused feature engineering)
- All src/models/ (broken predictor)
- All src/data/ (unused loaders)
- All src/api/ (incomplete)
- src/monitoring/ (placeholder)

**Result:** 
- Cleaner repo for hiring managers
- Easier to navigate
- No broken imports
- Professional appearance

**Risk:** LOW - These files aren't imported anywhere

**Benefit:** HIGH - Much better first impression
