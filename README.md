# EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm

EcoPredict Carbon là prototype nghiên cứu hỗ trợ ước lượng Product Carbon Footprint (PCF), benchmark theo ngành, giải thích yếu tố ảnh hưởng bằng Machine Learning và mô phỏng kịch bản phát thải theo định hướng LCA/ISO. Hệ thống dùng để hỗ trợ phân tích sơ bộ, không thay thế kiểm kê LCA chính thức hoặc chứng nhận ISO/EPD.

## 1. File chính

- `app.py`: giao diện Streamlit.
- `carbon_utils.py`: xử lý dữ liệu, feature engineering, model pipeline, LCA bottom-up, OOD/confidence.
- `scenario_projection.py`: mô phỏng kịch bản tham số, không dùng ARIMA.
- `train_advanced_models.py`: huấn luyện classification/regression, class imbalance handling, SHAP, sensitivity, lưu model.
- `imbalance_handler.py`: phân phối lớp, class weights, SMOTE/fallback, diagnostics lớp High.
- `model_interpretation.py`: helper SHAP/XAI.
- `generate_shap_explanations.py`: sinh lại SHAP plots từ model package.
- `hyperparameter_tuning.py`: GridSearchCV dùng `f1_macro` và SMOTE pipeline cho classification.
- `sensitivity_analysis.py`: tornado chart, heatmap 2 chiều, scenario sensitivity.
- `tests/`: unit tests cho core functions.
- `requirements.txt`: thư viện cần cài.
- `carbon_catalogue.csv`, `data/`: dữ liệu đầu vào.
- `outputs/`: metric, hình, bảng, model đã train.

## 2. Chạy trên Windows PowerShell

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Nếu môi trường đã có `.venv`, có thể chạy nhanh:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 3. Lệnh bổ sung

Sinh lại model và toàn bộ output:

```powershell
python train_advanced_models.py
```

Sinh lại SHAP plots:

```powershell
python generate_shap_explanations.py
```

Chạy GridSearchCV:

```powershell
python hyperparameter_tuning.py
```

Sinh sensitivity plots:

```powershell
python sensitivity_analysis.py
```

Chạy unit tests:

```powershell
python -m pytest tests -v --cov=carbon_utils --cov=scenario_projection --cov-report=term-missing
```

## 4. Tổng quan hệ thống

### 4.1 Mục tiêu hệ thống

EcoPredict Carbon là hệ thống hỗ trợ dự báo và phân tích phát thải carbon của sản phẩm dựa trên dữ liệu Product Carbon Footprint (PCF) và mô hình học máy.

Hệ thống giúp người dùng ước lượng nhanh mức phát thải, phân loại sản phẩm theo nhóm Low, Medium, High và so sánh kết quả với dữ liệu tham chiếu.

### 4.2 Phạm vi sử dụng

Hệ thống được xây dựng như một prototype nghiên cứu và hỗ trợ phân tích sơ bộ.

EcoPredict Carbon không thay thế kiểm kê LCA chính thức, chứng nhận ISO hoặc công bố EPD.

### 4.3 Nguồn dữ liệu

Hệ thống sử dụng các nguồn dữ liệu chính:

- Carbon Catalogue: dữ liệu PCF lịch sử.
- OpenPCF by Terralytiq: nguồn dữ liệu chính để mở rộng mẫu sản phẩm/vật liệu.
- Open CEDA: nguồn tham chiếu hệ số phát thải theo ngành và quốc gia.

Việc tập trung vào OpenPCF giúp tăng số lượng mẫu, đặc biệt là nhóm sản phẩm phát thải cao.

### 4.4 Chức năng chính

Hệ thống hỗ trợ các chức năng:

- Dự báo giá trị PCF của sản phẩm.
- Phân loại mức phát thải Low, Medium, High.
- So sánh với benchmark theo ngành.
- Phân tích yếu tố ảnh hưởng đến kết quả bằng SHAP/XAI.
- Mô phỏng kịch bản giảm phát thải trong tương lai.
- Hiển thị kết quả bằng giao diện web Streamlit.

### 4.5 Mô hình học máy

Hệ thống sử dụng các mô hình học máy cho hai bài toán chính:

- Phân loại mức phát thải.
- Hồi quy giá trị PCF.

Các chỉ số đánh giá gồm accuracy, F1-macro, balanced accuracy, recall theo từng lớp, MAE, RMSE và R².

### 4.6 Xử lý mất cân bằng lớp

Hệ thống bổ sung các kỹ thuật xử lý mất cân bằng dữ liệu như class weight và SMOTE.

Ngoài ra, hệ thống theo dõi riêng các chỉ số recall_high, f1_high và high_class_warning để kiểm tra khả năng nhận diện sản phẩm phát thải cao.

### 4.7 Giải thích mô hình

Hệ thống có các công cụ giải thích kết quả như:

- Permutation Importance.
- SHAP feature ranking.
- SHAP beeswarm.
- SHAP waterfall.
- Sensitivity analysis.

Các công cụ này giúp người dùng hiểu yếu tố nào ảnh hưởng mạnh đến kết quả dự báo.

### 4.8 Tuning và kiểm thử

Hệ thống bổ sung:

- GridSearchCV ở chế độ nhẹ để so sánh mô hình mặc định và mô hình tinh chỉnh.
- Unit tests cho các hàm lõi trong `carbon_utils.py` và `scenario_projection.py`.
- Bảng kết quả tuning và test được lưu trong `outputs/tables/`.

### 4.9 Công nghệ sử dụng

Hệ thống được xây dựng bằng Python và Streamlit.

Các thư viện chính gồm Pandas, NumPy, Scikit-learn, Imbalanced-learn, SHAP, Plotly, Matplotlib, Pytest và Joblib.

## 5. Phạm vi sử dụng đúng

Phù hợp để:

- Ước lượng sơ bộ PCF.
- Benchmark sản phẩm với trung vị ngành.
- Giải thích yếu tố ảnh hưởng.
- Mô phỏng kịch bản giảm phát thải.
- Hỗ trợ học thuật và ra quyết định sơ bộ.

Chưa phù hợp để:

- Chứng nhận ISO/EPD chính thức.
- Khai báo ESG/green claims có giá trị pháp lý.
- Thay thế LCA chính thức với physical data collection và critical review.

## 6. Ghi chú học thuật

Nếu metric như accuracy hoặc ROC AUC rất cao, không nên hiểu là mô hình hoàn hảo. Với bài toán này, nhãn Low/Medium/High được xây dựng từ ngưỡng PCF, đồng thời một số đặc trưng có liên quan đến emission factor. Vì vậy cần ưu tiên xem `F1-macro`, `Balanced Accuracy`, `recall_high`, confusion matrix và kiểm thử hold-out theo sản phẩm/quốc gia/thời gian.

## 7. Bản v8 OpenPCF-focused

Bản này đã được chỉnh để tập trung vào OpenPCF thay vì chỉ lấy mẫu nhỏ 1.000 dòng như phiên bản trước.

- OpenPCF hợp lệ: khoảng 22.886 dòng.
- Tập huấn luyện thực tế dùng stratified sample 12.000 dòng để chạy ổn trên máy cá nhân/Streamlit Cloud.
- Phân phối nhãn sau split:
  - Train: Low 2463, Medium 4755, High 2382.
  - Test: Low 616, Medium 1188, High 596.
- Mô hình phân loại tốt nhất: Random Forest + High-threshold tuned.
- Recall High trên test: 1.0 trong bản split OpenPCF-focused.
- Mô hình hồi quy tốt nhất: Extra Trees Regressor.

Lưu ý học thuật: kết quả tốt hơn vì test đã có đủ mẫu High và dữ liệu OpenPCF đồng nhất hơn theo đơn vị kgCO2e/kg. Đây vẫn là prototype hỗ trợ phân tích sơ bộ, không thay thế kiểm kê LCA chính thức hoặc chứng nhận ISO/EPD.

Lệnh train lại bản OpenPCF-focused:

```powershell
python train_advanced_models.py
python -m streamlit run app.py
```

Nếu cần sinh lại SHAP sau khi train:

```powershell
python generate_shap_explanations.py
```
