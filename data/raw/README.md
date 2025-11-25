# Raw Data Directory

## 📁 Purpose
This directory contains the **raw, unprocessed dataset** for the Abu Dhabi Property Price Prediction project.

## 📥 Dataset Placement

**Place your new dataset here with the filename:**
```
abu_dhabi_properties_raw.csv
```

## 📋 Expected Dataset Format

Your CSV file should contain the following columns (at minimum):

### Required Columns:
- `price` - Monthly rental price in AED (float)
- `bedrooms` - Number of bedrooms (int or "Studio")
- `bathrooms` - Number of bathrooms (int)
- `size` - Property size in square feet (float)
- `location` - Region/area name (string)
- `property_type` - Type of property: Apartment, Villa, Townhouse, Penthouse (string)

### Optional Columns:
- `property_id` - Unique identifier (int)
- `furnished` - Furnished status (boolean or string)
- `url` - Property listing URL (string)
- `scraped_at` - Timestamp of data collection (datetime)

## ⚠️ Important Notes

1. **Do NOT modify** the raw dataset after placing it here
2. All data cleaning and transformations will be done in the `data/processed/` directory
3. Keep this as the **single source of truth** for your original data
4. If you need to update the dataset, replace the entire file (don't edit in place)

## 🚫 What NOT to Place Here

- Processed/cleaned datasets
- Feature-engineered datasets
- Train/test splits
- Model outputs
- Temporary files

## ✅ Ready to Proceed

Once you place `abu_dhabi_properties_raw.csv` here, you can proceed with:
1. Data cleaning (→ `data/processed/`)
2. Feature engineering (→ `data/features/`)
3. Model training (→ `model_outputs/`)

---

**Current Status:** Directory is clean and ready for your new dataset! 🎯
