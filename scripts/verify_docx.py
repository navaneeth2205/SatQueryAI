import os
import docx

def verify():
    path = "METHODOLOGY.docx"
    assert os.path.exists(path), "File does not exist"
    doc = docx.Document(path)
    print(f"File size: {os.path.getsize(path):,} bytes")
    print(f"Paragraphs: {len(doc.paragraphs)}")
    print(f"Tables: {len(doc.tables)}")

    # Check images
    images = []
    for i, p in enumerate(doc.paragraphs):
        blips = p._p.xpath('.//a:blip')
        if blips:
            images.append((i, p.text))
    print(f"Found {len(images)} image paragraphs:")
    for idx, txt in images:
        print(f"  Paragraph {idx}: {txt}")

    # Check sections 1 to 10
    required_sections = [
        "1. Problem and Objectives",
        "2. Related Work",
        "3. Dataset",
        "4. Data Preprocessing",
        "5. System Architecture / Flow Diagram",
        "6. Mathematical Formulation",
        "7. Proposed Algorithm",
        "8. Preliminary Results",
        "9. Comparative Study",
        "10. References"
    ]
    
    found_sections = []
    for p in doc.paragraphs:
        for s in required_sections:
            if p.text.strip().startswith(s):
                found_sections.append(s)
    
    print("\nRequired Sections Check:")
    for s in required_sections:
        if s in found_sections:
            print(f"  [PASS] {s}")
        else:
            print(f"  [FAIL] Missing {s}")

    # Check Table 1 - Student Info
    t0 = doc.tables[0]
    print(f"\nStudent Team Table ({len(t0.rows)} rows):")
    for r in t0.rows:
        row_txt = [c.text.strip() for c in r.cells]
        print(f"  {row_txt}")

    # Check Tables
    print(f"\nAll Tables Count: {len(doc.tables)}")
    for i, t in enumerate(doc.tables):
        headers = [c.text.strip().replace('\n', ' ') for c in t.rows[0].cells]
        print(f"  Table {i+1}: {len(t.rows)} rows x {len(t.columns)} cols | Header: {headers[:2]}")

if __name__ == "__main__":
    verify()
