import PyPDF2
import re
import fitz  # PyMuPDF
import os
import pdfplumber
import pandas as pd

def extract_images_from_pdf(pdf_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate over each page
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        images = page.get_images(full=True)

        # Iterate over each image
        for image_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Save the image
            image_filename = f"{output_folder}/image_{page_index+1}_{image_index+1}.{image_ext}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)

            print(f"Saved image: {image_filename}")


def read_pdf_with_pypdf2(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text

def extract_abstract(text):
    # Use regex to find the start of the abstract
    abstract_start = re.search(r'\bAbstract\b|\bABSTRACT\b', text)
    if not abstract_start:
        return "Abstract not found"

    # Extract text starting from the abstract
    start_index = abstract_start.end()
    text_after_abstract = text[start_index:].strip()
    
    # Find the end of the abstract (assuming next section starts with a title or a keyword)
    abstract_end = re.search(r'\n[A-Z ]{2,}\n', text_after_abstract)
    if abstract_end:
        end_index = abstract_end.start()
        abstract_text = text_after_abstract[:end_index].strip()
    else:
        abstract_text = text_after_abstract.strip()  # If no clear end is found, take all the remaining text
    
    return abstract_text


def extract_tables_from_pdf(pdf_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            # Iterate over extracted tables
            for table_num, table in enumerate(tables):
                # Convert table to DataFrame
                df = pd.DataFrame(table)
                
                # Save DataFrame to CSV
                csv_filename = f"{output_folder}/table_page_{page_num+1}_table_{table_num+1}.csv"
                df.to_csv(csv_filename, index=False)

                print(f"Saved table: {csv_filename}")

def remove_references(text):
    # Use regex to find the start of the references section
    references_start = re.search(r'\bReferences\b|\bREFERENCES\b|\bBibliography\b|\bBIBLIOGRAPHY\b', text)
    if not references_start:
        return text

    # Extract text before the references section
    end_index = references_start.start()
    text_before_references = text[:end_index].strip()

    return text_before_references

def main():
    file_path = '/Users/yangz/Documents/projects/llm4catalyst/documents/gqd.pdf'
    pdf_text = read_pdf_with_pypdf2(file_path)
    
    # Remove references
    text_without_references = remove_references(pdf_text)
    
    # Extract abstract
    abstract = extract_abstract(text_without_references)
    
    print("Abstract:\n", abstract)

    output_folder = '/Users/yangz/Documents/projects/llm4catalyst/extracted_images'
    extract_images_from_pdf(file_path, output_folder)
    output_folder = '/Users/yangz/Documents/projects/llm4catalyst/extracted_tables'
    extract_tables_from_pdf(file_path, output_folder)


if __name__ == "__main__":
    main()
