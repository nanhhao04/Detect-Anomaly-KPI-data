# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from MLDetectTool import plot_anomaly, get_window_averages,extract_anomaly_info,create_json_output,remove_file_safe


path1 = "data/data_hl19_1.csv"
path2 = "data/data_hl19_2.csv"
data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
data_avg1 = get_window_averages(data1, window_len=1)
data_avg2 = get_window_averages(data2, window_len=1)
data_avg1["node"] = 1
data_avg2["node"] = 2
data_avg = pd.concat([data_avg1, data_avg2])
print("Data trung bình đã load:", data_avg.shape)

arial_path = r"C:\Windows\Fonts\arial.ttf"
if not os.path.exists(arial_path):
    raise FileNotFoundError(f"Không tìm thấy font Arial tại: {arial_path}")
pdfmetrics.registerFont(TTFont("Arial", arial_path))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Vietnamese", fontName="Arial", fontSize=11, leading=14))

# Sinh file pdf với đầu vào là file json
def create_pdf_output(answer_text, json_input="result_structured.json", output_pdf="result.pdf"):
    anomalies = extract_anomaly_info(json_input)

    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    content = []
    content.append(Paragraph("<b>PHÂN TÍCH BẤT THƯỜNG MẠNG</b>", styles["Vietnamese"]))
    content.append(Spacer(1, 0.5 * cm))
    content.append(Paragraph("<b>Kết quả phân tích:</b>", styles["Vietnamese"]))
    content.append(Paragraph(answer_text.replace("\n", "<br/>"), styles["Vietnamese"]))
    content.append(Spacer(1, 0.5 * cm))

    # --- Thêm biểu đồ cho mỗi anomaly ---
    if not anomalies.empty:
        content.append(Paragraph("<b>Biểu đồ minh họa:</b>", styles["Vietnamese"]))
        for _, row in anomalies.iterrows():
            path = plot_anomaly(data_avg, row["date"], row["node"], row["field"])
            if path:
                content.append(Image(path, width=14 * cm, height=6 * cm))
                content.append(Spacer(1, 0.4 * cm))

    doc.build(content)
    print(f"Đã tạo file PDF: {output_pdf}")
    return output_pdf

class SimpleRetriever:
    def __init__(self, docs, k=5):
        self.docs = docs
        self.k = k

    def get_relevant_documents(self, query: str):
        return self.docs[: self.k]

def build_documents(path_to_pdf):
    loader = PyPDFLoader(path_to_pdf)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return chunks


def answer_question_with_context(query: str, retriever: SimpleRetriever, llm: ChatOpenAI, context_prompt: str):
    docs = retriever.get_relevant_documents(query)
    response = llm.invoke([HumanMessage(content=context_prompt)])
    answer = getattr(response, "content", str(response))
    return answer, docs

if __name__ == "__main__":
    load_dotenv()

    pdf_path = "docs/Tai_lieu_cac_truong_du_lieu_chi_tiet.pdf"
    chunks = build_documents(pdf_path)
    retriever = SimpleRetriever(chunks, k=10)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # --- Prompt cho GPT ---
    with open("result_ml.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    subset = [d for d in data if d["date"].startswith("2025-02-07")]
    prompt = f"""
    Hãy phân tích lý do bất thường của mạng theo từng ngày, dựa trên dữ liệu KPI được cung cấp.

     Quy ước rất quan trọng:
    - Nếu trường `anomaly` = **-1** → đây là **điểm bất thường**.
    - Nếu `anomaly` = **1** → đây là **điểm bình thường**.
    - Một ngày có thể có **nhiều khoảng thời gian bất thường liên tiếp**, hãy gộp lại khi phân tích.
    - Với mỗi khoảng bất thường, chỉ cần nêu **1 đến 3 trường dữ liệu nổi bật nhất** (có giá trị cao hoặc thay đổi lớn).

    Yêu cầu đầu ra theo định dạng:
    Phân tích bất thường mạng ngày <YYYY-MM-DD HH:MM:SS> - (nodeX) - (score: ...)
    Trường dữ liệu nguyên nhân gây ra bất thường là ... (liệt kê 1–3 trường có giá trị và ý nghĩa)
    Nguyên nhân có thể là do ...

    Nếu có nhiều khoảng bất thường trong cùng ngày, hãy liệt kê từng khoảng riêng biệt.

    Dữ liệu:
    {json.dumps(subset, ensure_ascii=False, indent=2)}
    """

    query = "Hãy phân tích lý do bất thường của ngày."
    answer, used_docs = answer_question_with_context(query, retriever, llm, prompt)
    create_json_output(answer, json_path="result_ml.json", output_path="result_structured.json")
    create_pdf_output(answer_text=answer, json_input="result_structured.json", output_pdf="docs/result.pdf")
    remove_file_safe(path="result_structured.json")