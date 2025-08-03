# 🎓 Fuzzy-Based School Recommendation System

This project is a **web-based school recommendation system** using fuzzy logic, built with **Python + Flask + scikit-fuzzy**. It helps users find the most suitable schools around them based on distance, facilities, and extracurricular preferences.

> ✅ Note: The dataset currently only contains a selection of schools located in **Malang City, Indonesia**.

---

## 🚀 Features

- 📍 **Automatic location detection** using browser Geolocation API
- 🎯 **User preference weighting**: users can assign importance levels to facilities, extracurriculars, or proximity
- 🔎 **Fuzzy logic scoring**:
  - Matches school features with user preferences
  - Computes scores using fuzzy rules
- 📊 **Sorted recommendation results** with scores and distances
- 🖼️ **Detail page for each school**, including images and feature highlights
- ⚡ Built entirely with Flask (no external APIs required)

---

## 🧠 How It Works

1. **User inputs preferences**:
   - School level (elementary, junior high, senior high)
   - Desired facilities and extracurriculars
   - Importance level (0–5) for facilities, activities, and distance

2. **System calculates scores**:
   - Uses fuzzy logic to assess:
     - Facility match (how many requested facilities exist)
     - Extracurricular match
     - Distance (in km from current location)
   - Rules defined using `skfuzzy.control` map each input to a final suitability score

3. **Recommendations displayed**:
   - Top matching schools appear sorted by score
   - Users can view more details and images for each result

---

## 📺 Demo Video

[Demo Video(https://youtu.be/QReEand8NGs?si=HiHnAFrgo7bSfp8a)



