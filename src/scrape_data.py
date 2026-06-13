"""
STEP 1: DATA COLLECTION
=======================
This script downloads Orbia's public documents (web pages, PDFs) 
so we can later build a knowledge base for the chatbot.

What it does:
- Fetches HTML pages from Orbia's website
- Downloads PDF reports (annual reports, investor presentations)
- Saves everything as text files in data/raw/
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import time

# ─── CONFIGURATION ───────────────────────────────────────────
# Where to save raw data
RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ─── HELPER: Extract text from a webpage ─────────────────────
def scrape_webpage(url):
    """
    Fetch a URL, parse the HTML, extract visible text.
    Returns the text content as a string.
    
    How it works:
    1. requests.get() downloads the raw HTML
    2. BeautifulSoup parses the HTML into a tree structure
    3. We find all <p>, <h1>-<h6>, <li>, <span> tags (common text containers)
    4. We extract just the text from each tag and combine it
    """
    try:
        # Send HTTP GET request to the URL
        # headers mimic a real browser so the server doesn't block us
        response = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        # Raise an error if status code is not 200 (OK)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup using the 'html.parser'
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements (they contain code, not content)
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()  # remove the element from the tree
        
        # Find all text-containing tags
        text_parts = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "span"]):
            # Get the text, strip whitespace
            text = tag.get_text(strip=True)
            if text:  # Only add non-empty text
                text_parts.append(text)
        
        # Join all text parts with newlines
        return "\n".join(text_parts)
    
    except Exception as e:
        print(f"  [ERROR] Failed to scrape {url}: {e}")
        return ""

# ─── HELPER: Download a PDF ──────────────────────────────────
def download_pdf(url, filename):
    """
    Download a PDF file from a URL and save it locally.
    
    How it works:
    1. Stream the PDF content in chunks (so we don't load the whole file in memory)
    2. Write each chunk to a file
    """
    try:
        response = requests.get(url, timeout=30, stream=True, headers={
            "User-Agent": "Mozilla/5.0"
        })
        response.raise_for_status()
        
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "wb") as f:  # 'wb' = write binary mode (for PDFs)
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                f.write(chunk)
        print(f"  [OK] Downloaded PDF: {filename}")
        return filepath
    
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return None

# ─── SCRAPE ORBIA'S KEY PAGES ────────────────────────────────
def scrape_orbia_pages():
    """
    Scrape the most important Orbia pages for our knowledge base.
    
    These pages cover:
    - The company overview
    - Each of the 5 business groups
    - Careers information
    - Investor relations
    
    Each page is saved as a .txt file in data/raw/
    """
    
    # List of (url, filename) pairs to scrape
    pages = [
        ("https://www.orbia.com/this-is-orbia/", "orbia_overview.txt"),
        ("https://www.orbia.com/this-is-orbia/business-groups/", "orbia_business_groups.txt"),
        ("https://www.orbia.com/careers/", "orbia_careers.txt"),
        ("https://www.orbia.com/careers/how-we-hire/", "orbia_how_we_hire.txt"),
        ("https://www.orbia.com/investor-relations/", "orbia_investor_relations.txt"),
    ]
    
    print("=" * 60)
    print("SCRAPING ORBIA WEB PAGES")
    print("=" * 60)
    
    for url, filename in pages:
        print(f"\nFetching: {url}")
        text = scrape_webpage(url)
        
        if text:
            filepath = os.path.join(RAW_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:  # 'w' = write text mode
                f.write(f"SOURCE: {url}\n{'='*60}\n")
                f.write(text)
            print(f"  [SAVED] {filename} ({len(text)} characters)")
        else:
            print(f"  [SKIPPED] No content extracted")
        
        # Small delay to be polite to the server
        time.sleep(1)

# ─── DOWNLOAD PDF REPORTS ────────────────────────────────────
def download_pdfs():
    """
    Download Orbia's annual reports and investor presentations as PDFs.
    These contain rich detailed content about strategy, finances, and operations.
    
    Note: PDF URLs may change over time. If a download fails,
    you can manually download PDFs from https://www.orbia.com/investor-relations/
    and place them in data/raw/
    """
    
    print("\n" + "=" * 60)
    print("DOWNLOADING PDF REPORTS")
    print("=" * 60)
    
    # List of (pdf_url, local_filename) pairs
    pdfs = [
        # Corporate overview - this URL was found during research
        ("https://www.orbia.com/49a013/siteassets/5.-investor-relations/presentations/orbia-corporate-overview-2025.pdf",
         "orbia_corporate_overview_2025.pdf"),
    ]
    
    for url, filename in pdfs:
        print(f"\nDownloading: {url}")
        download_pdf(url, filename)
        time.sleep(1)

# ─── CREATE SIMPLE TEXT FILES FROM KEY INFO ──────────────────
def create_manual_docs():
    """
    Create text files from the key Orbia information we gathered during research.
    This ensures the chatbot has high-quality data even if PDF downloads fail.
    
    These files cover:
    1. Company snapshot (employees, revenue, leadership)
    2. The five business groups in detail
    3. Digital transformation / Wavin iConnect platform
    4. Values and culture
    """
    
    print("\n" + "=" * 60)
    print("CREATING MANUAL DOCUMENTS FROM RESEARCH DATA")
    print("=" * 60)
    
    # ── File 1: Company Overview ──
    overview_text = """SOURCE: Orbia Company Research
============================

COMPANY SNAPSHOT:
- Name: Orbia Advance Corporation, S.A.B. de C.V.
- Purpose: "To advance life around the world"
- CEO: Sameer S. Bharadwaj
- Employees: 23,000+
- Sales (2025): $7.6 Billion
- EBITDA (2025): $1.1 Billion
- Operations in: 50+ countries
- Sales in: 100+ countries
- Headquarters: Boston, USA / Mexico City, Mexico / Amsterdam, Netherlands / Tel Aviv, Israel
- Industry: Basic and advanced materials, specialty products, innovative solutions

BUSINESS SECTORS:
1. Polymer Solutions (Vestolit & Alphagary) - PVC resins, compounds, specialty polymers
2. Building & Infrastructure (Wavin) - Water management, pipes, fittings, smart building solutions
3. Precision Agriculture (Netafim) - Drip irrigation, precision farming, digital agriculture
4. Connectivity Solutions (Dura-Line) - Telecommunications conduit, HDPE products, fiber infrastructure
5. Fluor & Energy Materials (Koura) - Fluorspar mining, refrigerants, propellants, battery materials

GLOBAL CHALLENGES ORBIA ADDRESSES:
- Food and water security
- Connectivity and information access
- Climate resilience and decarbonization

VALUES:
- Bravery: Courage to innovate and challenge the status quo
- Responsibility: Own our impact on people and planet
- Diversity: Embrace different perspectives to drive better outcomes
"""
    
    with open(os.path.join(RAW_DIR, "orbia_company_snapshot.txt"), "w", encoding="utf-8") as f:
        f.write(overview_text)
    print("  [OK] Created orbia_company_snapshot.txt")
    
    # ── File 2: Business Groups Deep Dive ──
    business_text = """SOURCE: Orbia Business Groups Research
=====================================

1. POLYMER SOLUTIONS (Brands: Vestolit, Alphagary)
   - Products: PVC general resins, specialty resins, compounds, derivatives, additives
   - Applications: Pipes for drinking water, sterile healthcare equipment, construction materials
   - Also produces: Plasticizers, TPE, TPU, CPE, EVA compounds
   - Key advantage: Vertical integration secures supply and enables innovation
   - Serves: Infrastructure, health & well-being, automotive, construction industries

2. BUILDING & INFRASTRUCTURE (Brand: Wavin)
   - About: Innovative solutions for global building and infrastructure industry
   - 70+ years of product development experience
   - 10,000+ employees across 43 production sites
   - Serves 80+ countries
   - Key solutions:
     * Water distribution (clean drinking water)
     * Sanitation systems
     * Climate-resilient cities (stormwater management)
     * Smart building solutions (indoor climate control)
     * Blue-green roofs, tree tanks
   - Digital: Wavin iConnect platform - smart connected building solutions
   - Target: 100,000+ connected devices
   - Focus: Urban climate resilience, water management, energy efficiency

3. PRECISION AGRICULTURE (Brand: Netafim)
   - Founded: 1965
   - Claim: World's largest irrigation company
   - Pioneered the drip irrigation revolution
   - 4,500+ employees worldwide
   - 33 subsidiaries, 18 manufacturing plants
   - Services: End-to-end solutions from water source to root zone
   - Products: Drip irrigation, fertigation, greenhouses, digital farming
   - Digital: Real-time monitoring, analysis, automated control
   - Reaches millions of farmers in 100+ countries
   - Slogan: "Grow more with less"

4. CONNECTIVITY SOLUTIONS (Brand: Dura-Line)
   - Products: Telecommunications conduit, cable-in-conduit, HDPE products
   - Production: 500+ million meters of connectivity infrastructure per year
   - Creates physical pathways for fiber and network technologies
   - Connects cities, homes, and people worldwide
   - Serves largest telecom and data providers globally

5. FLUOR & ENERGY MATERIALS (Brand: Koura)
   - Holds: World's largest fluorspar mine
   - Products: Fluorspar, fluorine intermediates, refrigerants, propellants
   - Applications: Automotive, infrastructure, semiconductor, health, medicine
   - Also: Climate control, food cold chain, energy storage, computing, telecom
   - Key focus: Advanced battery materials for North American supply chain
   - Low global warming potential refrigerants
"""
    
    with open(os.path.join(RAW_DIR, "orbia_business_deep_dive.txt"), "w", encoding="utf-8") as f:
        f.write(business_text)
    print("  [OK] Created orbia_business_deep_dive.txt")
    
    # ── File 3: Digital Transformation ──
    digital_text = """SOURCE: Orbia Digital Transformation Research
============================================

WAVIN iCONNECT PLATFORM:
- Smart building platform running on Google Cloud
- Powers: Smart building, energy management, urban climate resilience, water management
- Target: 100,000+ connected devices across Europe and beyond
- Technology stack: Google Cloud (BigQuery, Spanner, Pub/Sub, Cloud Run)
- Backend language: Go
- Architecture: Distributed, cloud-native, microservices
- Features: Device-to-cloud communication, IoT protocols, device provisioning
- Development practices: Agile, CI/CD, testing, DevOps
- Focus areas: Indoor climate control, sustainable water management

ORBIA'S TECHNOLOGY FOCUS:
- Digital farming: Real-time monitoring, automated irrigation control
- Smart city solutions: Blue-green roofs, stormwater management
- Connected building systems: Heating, cooling, energy management
- IoT infrastructure: Sensors, connected devices, data analytics
- Digital transformation across all 5 business groups
- Innovation in materials science for sustainability

SUSTAINABILITY & ESG:
- ImpactMark: ESG performance indicators published transparently
- Focus: Circular economy, decarbonization, water stewardship
- UN Sustainable Development Goals alignment
- 2 recycling facilities
- Low GWP refrigerants
- Battery materials for clean energy
"""
    
    with open(os.path.join(RAW_DIR, "orbia_digital_transformation.txt"), "w", encoding="utf-8") as f:
        f.write(digital_text)
    print("  [OK] Created orbia_digital_transformation.txt")

# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    When this script is run directly (not imported), execute all three steps:
    1. Scrape web pages
    2. Download PDFs
    3. Create manual research docs
    """
    scrape_orbia_pages()
    download_pdfs()
    create_manual_docs()
    
    # List all files we created
    print("\n" + "=" * 60)
    print("FILES IN data/raw/:")
    print("=" * 60)
    for f in sorted(os.listdir(RAW_DIR)):
        filepath = os.path.join(RAW_DIR, f)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {f:50s} ({size_kb:.1f} KB)")
    
    print("\n[DONE] Data collection complete!")
