# Capstone Project Poster

This directory contains the capstone project poster for the **Bangladeshi Traffic Sign Detection Using YOLOv11** project.

## 📄 Files

### Main Poster Files

1. **`CAPSTONE_PROJECT_POSTER.md`** (20KB)
   - Complete poster content in Markdown format
   - Comprehensive project documentation
   - Academic-style presentation
   - Easy to read and edit

2. **`capstone_poster.html`** (26KB)
   - Interactive HTML poster
   - Professional web-based design
   - Responsive layout
   - Print-friendly
   - Can be opened in any web browser

3. **`poster_latex.tex`** (12KB)
   - LaTeX/beamer poster template
   - Professional academic poster format
   - A0 size (standard conference poster)
   - Requires LaTeX compilation

4. **`index.html`** (existing)
   - Previous poster version
   - Web-based research poster

5. **`style.css`** (existing)
   - Styling for index.html

## 🎨 Viewing the Posters

### HTML Poster (Recommended)

The easiest way to view the poster:

```bash
# Open in web browser
cd /home/mnx/bd-traffic-signs/poster
firefox capstone_poster.html
# or
google-chrome capstone_poster.html
# or
xdg-open capstone_poster.html
```

**Features:**
- ✅ Professional design with gradients and colors
- ✅ Interactive sections (hover effects)
- ✅ Mobile responsive
- ✅ Print-ready (Ctrl+P to print)
- ✅ No compilation needed

### Markdown Poster

View in any text editor or Markdown viewer:

```bash
# View in terminal
cat CAPSTONE_PROJECT_POSTER.md

# Or open in editor
nano CAPSTONE_PROJECT_POSTER.md
code CAPSTONE_PROJECT_POSTER.md
```

### LaTeX Poster

Requires LaTeX installation:

```bash
# Install LaTeX (if not installed)
sudo apt-get install texlive-full

# Compile the poster
cd /home/mnx/bd-traffic-signs/poster
pdflatex poster_latex.tex

# This generates: poster_latex.pdf
```

## 📊 Poster Content

The poster includes:

### 1. Header Section
- Project title
- Subtitle
- Author information
- Institution details
- Key achievement badges

### 2. Abstract
- Project overview
- Key achievements
- Keywords

### 3. Introduction & Motivation
- Problem statement
- Background
- Objectives

### 4. Dataset (BRSDD)
- Dataset statistics (8,953 images, 29 classes)
- Data distribution
- All 29 traffic sign categories
- Source and format information

### 5. Methodology
- YOLOv11 architecture diagram
- Why YOLOv11?
- Training configuration
- Loss functions
- Data augmentation techniques

### 6. Results & Performance
- **Outstanding metrics:**
  - 99.45% mAP@50
  - 94.23% mAP@50-95
  - 97.91% Precision
  - 99.54% Recall
- Training progression table
- Inference speed benchmarks
- Model comparison

### 7. Deployment
- Android mobile application
- Web application (Gradio)
- Technical specifications

### 8. Applications
- Driver assistance systems
- Navigation systems
- Autonomous vehicles
- Educational tools

### 9. Conclusion & Future Work
- Key achievements
- Future enhancements
- Impact statement

### 10. References & Contact
- Academic references
- Dataset source
- Contact information
- Project statistics

## 🖨️ Printing Instructions

### HTML Poster

1. Open `capstone_poster.html` in a web browser
2. Press **Ctrl+P** (or Cmd+P on Mac)
3. In print dialog:
   - Set margins to "Minimum" or "None"
   - Enable "Background graphics"
   - Choose appropriate paper size (A3 or A4)
   - Select "Portrait" orientation
4. Print or "Save as PDF"

### LaTeX Poster

The LaTeX poster is designed for A0 size (standard conference poster):

```bash
# Compile to PDF
pdflatex poster_latex.tex

# This creates an A0-sized PDF
# Print at a poster printing service or scale down for smaller sizes
```

## 🎨 Customization

### HTML Poster

Edit `capstone_poster.html`:
- Colors: Modify CSS in `<style>` section
- Content: Edit HTML directly
- Layout: Adjust grid configurations

### Markdown Poster

Edit `CAPSTONE_PROJECT_POSTER.md`:
- Plain text format
- Easy to modify
- Can be converted to other formats

### LaTeX Poster

Edit `poster_latex.tex`:
- Professional academic format
- Requires LaTeX knowledge
- Highly customizable

## 📐 Poster Dimensions

- **HTML Poster:** Responsive (adapts to screen/print size)
- **LaTeX Poster:** A0 size (841mm × 1189mm)
- **PDF Export:** Customizable in print dialog

## 🚀 Quick Start

**To view the poster right now:**

```bash
cd /home/mnx/bd-traffic-signs/poster
firefox capstone_poster.html &
```

**To print to PDF:**

```bash
# Using wkhtmltopdf (if installed)
wkhtmltopdf capstone_poster.html capstone_poster.pdf

# Or use browser's print-to-PDF feature
```

## 📊 Poster Sections Summary

| Section | Content | Pages/Space |
|---------|---------|-------------|
| Header | Title, authors, badges | Top |
| Abstract | Project summary | 1 block |
| Problem | Motivation & objectives | 1 card |
| Dataset | BRSDD statistics | 2 cards |
| Methodology | YOLOv11 architecture | 1 large |
| Results | Performance metrics | 1 large |
| Deployment | Mobile + Web apps | 1 card |
| Applications | Real-world use cases | 1 large |
| Conclusion | Achievements & future | 1 large |
| Footer | References & contact | Bottom |

## 🎯 Key Highlights

The poster emphasizes:

✅ **Outstanding Performance:** 99.45% mAP@50  
✅ **Production-Ready:** Full deployment pipeline  
✅ **Comprehensive Dataset:** 8,953 images, 29 classes  
✅ **Multi-Platform:** Android + Web  
✅ **Research-Grade:** Publication ready  

## 📧 Contact

For questions about the poster:
- **Email:** capstone2024@northsouth.edu
- **GitHub:** github.com/nsu/bd-traffic-signs
- **Institution:** North South University, Bangladesh

---

**Created:** December 3, 2024  
**Version:** 1.0  
**Format:** Markdown, HTML, LaTeX  
**Status:** Ready for presentation
