from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class PdfSpiltter():
    def split(self, file_name, chunk_size=0, overlap_size=0):
        pdf_loader = PyMuPDFLoader(file_name)
        document_list = pdf_loader.load_and_split()

        print(len(document_list))
        print(type(document_list[0].page_content))
        # print(document_list[0].page_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = overlap_size,
            length_function = len,
            add_start_index = True
        )

        splitted_document_list = text_splitter.create_documents([document.page_content for document in document_list])
        return splitted_document_list

if __name__ == "__main__":
    pdf_splitter = PdfSpiltter()
    splitted_document_list = pdf_splitter.split("LangChain/llama2.pdf", 200, 100)
    for document in splitted_document_list:
        print(document.page_content)
        print('-------')