import pdfplumber

path_to_pdf = r"C:\Users\jimko\Desktop\ergasia\bmw_x1_ai_agent\data\bmw_manual.pdf"

all_text = []
with pdfplumber.open(path_to_pdf) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text.append(f"[Σελίδα {i+1}]\n{text}\n")
            
manual_text = "\n".join(all_text)

with open("manual_text_output.txt", "w", encoding="utf-8") as f:
    f.write(manual_text)
