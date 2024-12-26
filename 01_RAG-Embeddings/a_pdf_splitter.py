from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextBoxHorizontal

class PdfSplitter():
    def extract_text_from_pdf(self, filename, page_numbers=None, min_line_length=1):
        '''从 PDF 文件中（按指定页码）提取文字'''
        paragraphs = []
        buffer = ''
        full_text = ''
        for i, page_layout in enumerate(extract_pages(filename)):
            if page_numbers is not None and i not in page_numbers:
                continue
            for element in page_layout:
                # print(type(element))
                if isinstance(element, LTTextContainer):
                    text = element.get_text()
                    # print(text)
                    full_text += text + "\n"
            lines = full_text.split("\n")
            for line in lines:
                if len(line) >= min_line_length:
                    buffer = buffer + (" " + line) if not line.endswith("-") else line.strip("-")
                else:
                    paragraphs.append(buffer)
                    buffer = ""
        return [paragraph for paragraph in paragraphs if len(paragraph) > 0]

if __name__ == "__main__":
    pdf_splitter = PdfSplitter()
    result = pdf_splitter.extract_text_from_pdf("RAG-Embeddings/llama2.pdf", min_line_length=10)
    # print(result[:4])
    # print(result[1])
    # print(result[2])
    for paragraph in result[:4]:
        print(paragraph)