# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from MLDetectTool import (
    plot_anomaly,
    get_window_averages,
    extract_anomaly_info,
    create_json_output,
)

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




def create_pdf_output(answer_text, json_input="result_structured.json", output_pdf="result.pdf"):
    """
    Tạo báo cáo PDF tiếng Việt (Unicode) từ nội dung phân tích và dữ liệu anomaly.
    - Đọc danh sách anomaly từ file JSON (đã parse)
    - Sinh biểu đồ bằng hàm plot_anomaly(json_path, field, target_date, node)
    - Chèn phần mô tả (LLM answer_text) vào trước
    """
    # --- Đọc danh sách anomaly ---
    anomalies = extract_anomaly_info(json_input)
    if anomalies.empty:
        print(f"⚠️ Không tìm thấy anomaly trong {json_input}")
        return None

    # --- Tạo PDF ---
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    content = []

    # --- Tiêu đề ---
    content.append(Paragraph("<b>PHÂN TÍCH BẤT THƯỜNG MẠNG</b>", styles["Vietnamese"]))
    content.append(Spacer(1, 0.5 * cm))
    content.append(Paragraph("<b>Kết quả phân tích:</b>", styles["Vietnamese"]))

    # --- Làm sạch phần mô tả (bỏ JSON trong answer_text) ---
    filtered_text = []
    json_started = False
    for line in answer_text.splitlines():
        if line.strip().startswith("[") or line.strip().startswith("{"):
            json_started = True
        if not json_started:
            filtered_text.append(line)
    clean_answer_text = "\n".join(filtered_text)
    clean_answer_text = re.sub(r"```.*?```", "", answer_text, flags=re.DOTALL)
    clean_answer_text = re.sub(r"[*_#>`]+", "", clean_answer_text)
    clean_answer_text = re.sub(r"\n{2,}", "\n", clean_answer_text).strip()

    content.append(Paragraph(clean_answer_text.replace("\n", "<br/>"), styles["Vietnamese"]))
    content.append(Spacer(1, 0.5 * cm))

    # --- Phần biểu đồ ---
    content.append(Paragraph("<b>Biểu đồ minh họa các điểm bất thường:</b>", styles["Vietnamese"]))
    drawn = set()

    for _, row in anomalies.iterrows():
        field = row.get("field")
        node = row.get("node")
        date = row.get("date")

        # --- Bỏ qua giá trị thiếu ---
        if not field or not date:
            continue

        # Chuyển định dạng ngày
        try:
            target_day = pd.to_datetime(date).strftime("%Y-%m-%d")
        except Exception:
            continue

        key = (field, node, target_day)
        if key in drawn:
            continue

        # --- Gọi hàm plot_anomaly mới ---
        img_path = plot_anomaly(
            json_path="result_ml.json",  # file anomaly gốc (đầy đủ)
            field=field,
            full_data=data_avg,
            target_day=target_day,
            node=node,
            save_dir="plot_pics",
        )

        if isinstance(img_path, str) and os.path.exists(img_path):
            content.append(Image(img_path, width=14 * cm, height=6 * cm))
            content.append(Spacer(1, 0.4 * cm))
            drawn.add(key)

    # --- Xuất PDF ---
    doc.build(content)
    print(f"✅ Đã tạo file PDF: {output_pdf}")
    return output_pdf




def build_faiss_retriever(pdf_path, embedding_model, chunk_size=400, chunk_overlap=100, faiss_dir="faiss_index"):
    os.makedirs(faiss_dir, exist_ok=True)
    index_path = os.path.join(faiss_dir, "index.pkl")

    if os.path.exists(index_path):
        print(" Đang load FAISS index từ cache...")
        with open(index_path, "rb") as f:
            vectorstore = pickle.load(f)
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    print(f" Đang tạo FAISS index từ {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    with open(index_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print(" FAISS index đã được lưu cache.")
    return vectorstore.as_retriever(search_kwargs={"k": 5})


from huggingface_hub import InferenceClient
import os


def hf_infer(prompt: str) -> str:
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("Thiếu HUGGINGFACEHUB_API_TOKEN trong .env")
        return ""

    try:
        client = InferenceClient(model=model_id, token=token)
    except Exception as e:
        print("Không thể khởi tạo InferenceClient:", e)
        return ""

    messages = [{"role": "user", "content": prompt}]
    plain_prompt = prompt

    if hasattr(client, "chat_completion"):
        for kwargs in [
            dict(messages=messages, max_new_tokens=4096, temperature=0.3),
            dict(messages=messages, max_tokens=4096, temperature=0.3),
            dict(messages=messages, temperature=0.3)
        ]:
            try:
                print(f"Thử client.chat_completion(...) với kwargs={list(kwargs.keys())}")
                resp = client.chat_completion(**kwargs)
                try:
                    return resp.choices[0].message["content"]
                except Exception:
                    return getattr(resp, "generated_text", str(resp))
            except TypeError as e:
                print(f"Bỏ qua chat_completion cấu hình {list(kwargs.keys())}: {e}")
            except Exception as e:
                print("Lỗi khi gọi chat_completion:", e)

    if hasattr(client, "chat"):
        try:
            print("Thử client.chat(...)")
            resp = client.chat(messages=messages, max_new_tokens=1024, temperature=0.3)
            return getattr(resp, "generated_text", str(resp))
        except Exception as e:
            print("Lỗi khi gọi chat:", e)

    if hasattr(client, "text_generation"):
        try:
            print("Thử client.text_generation(...) (fallback cuối)")
            resp = client.text_generation(plain_prompt, max_new_tokens=1024, temperature=0.3)
            return getattr(resp, "generated_text", str(resp))
        except Exception as e:
            print("Lỗi khi gọi text_generation:", e)

    print("Không thể gọi model — có thể model này không hỗ trợ các API trên.")
    print("   ➜ Thử đổi sang model text2text như 'google/flan-t5-large' hoặc 'tiiuae/falcon-7b-instruct'.")
    return ""



def answer_question_with_context(query, retriever, context_prompt):
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])
    full_prompt = f"{context_prompt}\n\nTài liệu tham khảo:\n{context_text}\n\nCâu hỏi: {query}"
    answer = hf_infer(full_prompt)
    return answer, docs

if __name__ == "__main__":
    load_dotenv()

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pdf_path = "docs/Tai_lieu_cac_truong_du_lieu_chi_tiet.pdf"
    retriever = build_faiss_retriever(pdf_path, embedding_model)

    # --- Load dữ liệu anomaly ---
    with open("result_ml.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    subset = [d for d in data if d["date"].startswith("2025-02-07")]

    prompt = f"""
    Bạn là chuyên gia phân tích hiệu năng mạng viễn thông (KPI Analyst).

    Dữ liệu cung cấp dưới đây đã được hệ thống phát hiện là bất thường (tất cả các record đều là anomaly).
    Hãy dựa vào tài liệu KPI (FAISS retriever) và **Dữ liệu anomaly dưới đây** để viết **báo cáo phân tích bất thường trong ngày** theo yêu cầu sau:

    ---

    ### **Dữ liệu anomaly:**
    {json.dumps(subset, ensure_ascii=False, indent=2)}

    ---

    ### **Yêu cầu đầu ra:**

    Viết phần mô tả bằng tiếng Việt có cấu trúc như sau:

    **Dữ liệu được cung cấp cho <số lượng> điểm bất thường trong các khoảng thời gian **
    1. **Khoảng từ <hh:mm> đến <hh:mm> giờ ngày <dd/mm/yyyy>:** (lưu ý: ví dụ 08:00 đến 08:30 và không lặp lại)
       - Nêu rõ các trường KPI nổi bật trong khoảng này (tăng hoặc giảm bất thường).(khoảng 2 đến 4 trường)
       - Ghi rõ giá trị trung bình của từng trường KPI (ví dụ: "giá trị trung bình là 16.30%").
       - Nguyên nhân có thể xảy ra (ví dụ: "Tăng số lượng attach/service request dẫn đến tăng tải trên MME").
       - Điểm bất thường nhất <hh:mm>, có giá trị trường KPI nào thay đổi đột biến với giá trị là.
    2. ... (cho các khoảng khác tương tự)

    **Kết quả phân tích:**
    - Tổng hợp lại các trường KPI nổi bật nhất trong ngày và mô tả nguyên nhân ảnh hưởng chính.

    ---

    Sau phần mô tả, **xuất ra JSON hợp lệ** đúng định dạng sau (đặt trong khối ```json ... ```):

    ```json
    [
      {{
        "date": "<YYYY-MM-DDTHH:MM:SS>",
        "node": <node_id>,
        "field": "<tên KPI>",
        "reason": "<nguyên nhân tóm tắt>"
      }},
      ...
    ]
"""

    query = "Phân tích lý do bất thường trong ngày dựa theo KPI và tài liệu hướng dẫn."
    answer, used_docs = answer_question_with_context(query, retriever, prompt)
    print(answer)

    create_json_output(answer, json_path="result_ml.json", output_path="result_structured.json")
    create_pdf_output(answer_text=answer, json_input="result_structured.json", output_pdf="docs/result.pdf")

