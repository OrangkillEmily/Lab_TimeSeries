import pandas as pd
def load_and_prep_data(filepath: str, date_col: str = 'Month') -> pd.DataFrame:
    """
    1. Mục đích: 
       Đọc file CSV chứa dữ liệu chuỗi thời gian, ép kiểu cột thời gian về định dạng datetime 
       và thiết lập cột đó làm index cho DataFrame.
       
    2. Tham số đầu vào:
       - filepath (str): Đường dẫn tương đối hoặc tuyệt đối tới file CSV (VD: 'data/raw/airline-passengers.csv').
       - date_col (str): Tên cột chứa thông tin thời gian. Mặc định là 'Month'.
       
    3. Kết quả đầu ra:
       - pd.DataFrame: DataFrame đã được xử lý index thời gian, sẵn sàng cho bước EDA và Modeling.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    return df