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
    create_json_output
)

path = "data/data_hl19.csv"
path1 = "data/data_hl19_1.csv"
path2 = "data/data_hl19_2.csv"
full_data = pd.read_csv(path)
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
    # --- Đọc danh sách anomaly ---
    anomalies = extract_anomaly_info(json_input)
    if anomalies.empty:
        print(f" Không tìm thấy anomaly trong {json_input}")
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
    clean_answer_text = re.sub("JSON OUTPUT CUỐI CÙNG", "", clean_answer_text)
    clean_answer_text = re.sub(r"\n{2,}", "\n", clean_answer_text).strip()

    content.append(Paragraph(clean_answer_text.replace("\n", "<br/>"), styles["Vietnamese"]))
    content.append(Spacer(1, 0.5 * cm))

    # Phần biểu đồ
    content.append(Paragraph("<b>Biểu đồ minh họa các điểm bất thường:</b>", styles["Vietnamese"]))
    drawn = set()

    for _, row in anomalies.iterrows():
        field = row.get("field")
        node = row.get("node")
        date = row.get("date")

        # Bỏ qua giá trị thiếu
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
    print(f" Đã tạo file PDF: {output_pdf}")
    return output_pdf




def build_faiss_retriever(pdf_path, embedding_model, chunk_size=200, chunk_overlap=50, faiss_dir="faiss_index"):
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
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
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
    print("   ➜ Thử đổi sang model text2text khác.")
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

    import json

    # Giả sử `subset` là danh sách anomaly đã có trend
    json_text = json.dumps(subset, ensure_ascii=False, indent=2)

    prompt = f"""
    Bạn là chuyên gia phân tích hiệu năng mạng viễn thông (KPI Analyst). Nhiệm vụ của bạn là viết báo cáo phân tích bất thường dựa trên dữ liệu đầu vào.

    ---
    ###  Hướng dẫn sử dụng Tài liệu Tham khảo (FAISS Retriever):
    Bạn phải sử dụng Tài liệu tham khảo KPI được cung cấp để:
    1. Giải thích ý nghĩa KPI được liệt kê.
    2. Đưa ra nguyên nhân tiềm ẩn phù hợp với xu hướng bất thường (tăng/giảm).

    ---
    ###  Dữ liệu anomaly (bao gồm `window_start` và `window_end`):
    """ + json_text + """

    ---
    ##  QUY TẮC BẮT BUỘC VÀ LỌC TRÙNG LẶP (TUYỆT ĐỐI):

    1. **Gom Nhóm DUY NHẤT theo thời gian:** 
       - Bạn phải gom nhóm theo đúng cặp thời gian (`window_start`, `window_end`) được cung cấp trong dữ liệu.
       - Mỗi cặp DUY NHẤT chỉ được tạo **một (1) mục phân tích trong báo cáo văn bản.**
       - Bạn **KHÔNG ĐƯỢC tự thay đổi, tính toán lại, suy diễn hoặc làm tròn thời gian.**
       - Bạn phải sử dụng **CHÍNH XÁC** giá trị `window_start` và `window_end` được đưa ra trong dữ liệu.

    2. **Giới hạn số lượng mục phân tích:** 
       Số lượng mục phân tích trong văn bản **bằng đúng số khoảng thời gian DUY NHẤT**.

    3. **Xu hướng (`_trend`):**
       - KPI phải nêu rõ đang **tăng bất thường (`increasing`)** hoặc **giảm bất thường (`decreasing`)**.
       - Sử dụng trường `field_trend` trong dữ liệu, không tự suy luận khác.

    4. **Đồng bộ Văn bản và JSON OUTPUT:**
       - Mọi KPI được liệt kê trong báo cáo văn bản (tối đa 4 KPI/window) **phải xuất hiện đầy đủ trong JSON OUTPUT cuối cùng.**
       - **KHÔNG ĐƯỢC THIẾU bất kỳ KPI nào đã nêu trong phần văn bản.**

    5. **Chỉ sử dụng dữ liệu được cung cấp. Không tự bổ sung giá trị.**
    
    6. **Tất cả KPI trong báo cáo bắt buộc phải có:**
       - Giá trị KPI bất thường (số cụ thể từ dữ liệu).
       - Ý nghĩa KPI (dựa trên tài liệu tham khảo).

    7. **Đánh số thứ tự tự động theo từng khoảng thời gian.**

    ---
    ##  FORMAT BÁO CÁO (Văn bản) (CHỈ XUẤT WINDOW DUY NHẤT):

    **Sử dụng đánh số tự động (1., 2., 3.,...) cho các window duy nhất.**

    1. **Khoảng từ `<window_start>` đến `<window_end>`:**
       
       - Liệt kê 2–4 KPI theo mẫu bắt buộc:
       - `**<KPI>**: <Ý nghĩa từ tài liệu tham khảo> có giá trị <VALUE> và đang **<tăng/giảm> bất thường**.`
       - **Nguyên nhân:** 1–2 nguyên nhân phù hợp với hướng tăng/giảm.


    2. **<Tiếp tục đánh số tự động cho các window DUY NHẤT còn lại>**

    ---
    ##  JSON OUTPUT CUỐI CÙNG:

    - Tạo **1 record JSON cho MỌI KPI** được phân tích trong báo cáo văn bản.
    - Mỗi record phải sử dụng đúng timestamp từ **`window_start` hoặc `window_end`** (bạn chọn timestamp đại diện của cửa sổ là **`window_end`**).
    - Các trường bắt buộc:
      - `"date"`: sử dụng giá trị `window_end` (YYYY-MM-DDTHH:MM:SS)
      - `"node"`: node id
      - `"field"`: tên KPI
      - `"reason"`: mô tả nguyên nhân, **phải bao gồm xu hướng tăng/giảm.**

    **Ví dụ JSON đúng chuẩn:**
    ```json
    [
      {
        "date": "2025-02-07T20:20:00",
        "node": 2,
        "field": "SAU_4G",
        "reason": "SAU_4G tăng bất thường: có thể do lưu lượng ứng dụng tăng đột biến."
      },
      {
        "date": "2025-02-07T20:20:00",
        "node": 2,
        "field": "SERVICE_REQUEST_SR",
        "reason": "SERVICE_REQUEST_SR giảm bất thường: có thể do lỗi đồng bộ hóa."
      }
    ]
    ...

"""

    query = "Phân tích lý do bất thường trong ngày dựa theo KPI và tài liệu hướng dẫn."
    answer, used_docs = answer_question_with_context(query, retriever, prompt)
    print(answer)

    create_json_output(answer, json_path="result_ml.json", output_path="result_structured.json")
    create_pdf_output(answer_text=answer, json_input="result_structured.json", output_pdf="docs/result.pdf")

