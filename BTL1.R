# Load thư viện
library(sparklyr)
library(stats)
library(dplyr)
library(ggplot2)
library(lubridate)
library(readxl)
library(data.table)  # Tối ưu hiệu suất
library(cluster)
# Machine Learning - KMeans

# 1. Kết nối Spark ------------------------------------------------------
sc <- spark_connect(master = "local")

# 2. Đọc dữ liệu từ file ------------------------------------------------
file_path <- "C:/Users/user/Desktop/Bài tập lớn R/pseudo_facebook.csv"
fb_data <- spark_read_csv(sc, name = "fb_data", path = file_path, infer_schema = TRUE, header = TRUE)

# 3. Phân tích số lượng người dùng theo độ tuổi --------------------------
if ("age" %in% colnames(fb_data)) {
  fb_data %>%
    select(age) %>%
    collect() %>%
    ggplot(aes(x = age)) +
    geom_histogram(binwidth = 5, fill = "#1f77b4", color = "black", alpha = 0.7) +
    theme_minimal() +
    labs(title = "Phân bố độ tuổi người dùng", x = "Tuổi", y = "Số lượng")
}

# 4. Phân tích số lượng người dùng theo giới tính ----------------------
if ("gender" %in% colnames(fb_data)) {
  fb_data %>%
    group_by(gender) %>%
    summarise(N = n()) %>%
    collect() %>%
    ggplot(aes(x = gender, y = N, fill = gender)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = N), vjust = -0.5) +
    scale_fill_manual(values = c("male" = "blue", "female" = "pink")) +
    theme_minimal() +
    labs(title = "Số lượng người dùng theo giới tính", x = "Giới tính", y = "Số lượng")
}

# 5. Phân tích số lượng bài đăng theo ngày -----------------------------
if ("dob_day" %in% colnames(fb_data)) {
  fb_data %>%
    group_by(dob_day) %>%
    summarise(N = n()) %>%
    collect() %>%
    ggplot(aes(x = dob_day, y = N)) +
    geom_line(color = "#ff7f0e", size = 1) +
    theme_minimal() +
    labs(title = "Số lượng bài đăng theo ngày", x = "Ngày", y = "Số bài đăng")
}

# 6. Phân tích số lượng bài đăng theo năm -----------------------------
if ("dob_year" %in% colnames(fb_data)) {
  fb_data %>%
    group_by(dob_year) %>%
    summarise(N = n()) %>%
    collect() %>%
    ggplot(aes(x = dob_year, y = N)) +
    geom_line(color = "#0000FF", size = 1) +
    theme_minimal() +
    labs(title = "Số lượng bài đăng theo năm", x = "Năm", y = "Số bài đăng")
}

# 7. Phân tích số lượng bài đăng theo tháng -----------------------------
if ("dob_month" %in% colnames(fb_data)) {
  fb_data %>%
    group_by(dob_month) %>%
    summarise(N = n()) %>%
    collect() %>%
    ggplot(aes(x = dob_month, y = N)) +
    geom_line(color = "#FFFF00", size = 1) +
    theme_minimal() +
    labs(title = "Số lượng bài đăng theo tháng", x = "Tháng", y = "Số bài đăng")
}

# 8. Phân tích số lượng bạn bè của người dùng --------------------------
if ("friend_count" %in% colnames(fb_data)) {
  fb_data %>%
    select(friend_count) %>%
    collect() %>%
    ggplot(aes(x = friend_count)) +
    geom_histogram(binwidth = 50, fill = "#2ca02c", color = "black", alpha = 0.7) +
    theme_minimal() +
    labs(title = "Phân bố số lượng bạn bè", x = "Số bạn bè", y = "Số người dùng") +
    scale_x_continuous(limits = c(0, 5000))
}

# 9. Phân cụm người dùng theo độ tuổi và giới tính -------------------
if (all(c("age", "gender") %in% colnames(fb_data))) {
  # Chuẩn bị dữ liệu cho phân cụm
  fb_cluster <- fb_data %>%
    select(age, gender) %>%
    filter(age > 0, age < 100) %>%  # Lọc tuổi hợp lệ
    filter(gender %in% c("male", "female")) %>%  # Lọc giới tính hợp lệ
    mutate(gender_numeric = case_when(
      gender == "male" ~ 1.0,
      gender == "female" ~ 0.0
    )) %>%
    select(age, gender_numeric) %>%
    collect()  # Chuyển về R dataframe
  
  # Chuẩn hóa dữ liệu
  fb_cluster$age <- scale(fb_cluster$age)[,1]
  
  # Chuyển lại vào Spark
  fb_cluster_spark <- copy_to(sc, fb_cluster, "fb_cluster_spark", overwrite = TRUE)
  
  # Thực hiện phân cụm K-means
  tryCatch({
    kmeans_model <- fb_cluster_spark %>%
      ml_kmeans(
        features = c("age", "gender_numeric"),
        k = 3,
        seed = 123,
        init_mode = "random"
      )
    
    # Dự đoán cụm
    predictions <- ml_predict(kmeans_model, fb_cluster_spark) %>%
      collect()
    
    # Thêm lại thông tin gốc
    predictions$gender <- ifelse(predictions$gender_numeric == 1, "Nam", "Nữ")
    predictions$age <- predictions$age * sd(fb_data %>% select(age) %>% collect() %>% pull()) +
      mean(fb_data %>% select(age) %>% collect() %>% pull())
    
    # Vẽ biểu đồ phân cụm
    ggplot(predictions, aes(x = age, y = gender_numeric, color = factor(prediction))) +
      geom_jitter(alpha = 0.6, width = 0.3, height = 0.05) +
      scale_y_continuous(breaks = c(0, 1), labels = c("Nữ", "Nam")) +
      scale_color_manual(
        values = c("#E41A1C", "#377EB8", "#4DAF4A"),
        name = "Nhóm",
        labels = c("Nhóm 1", "Nhóm 2", "Nhóm 3")
      ) +
      labs(
        title = "Phân nhóm người dùng theo độ tuổi và giới tính",
        subtitle = paste("Số lượng mẫu:", nrow(predictions)),
        x = "Tuổi",
        y = "Giới tính"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "right"
      )
    
    # Tính toán thống kê cho từng nhóm
    group_stats <- predictions %>%
      group_by(prediction) %>%
      summarise(
        so_luong = n(),
        tuoi_tb = round(mean(age), 1),
        tuoi_min = round(min(age), 1),
        tuoi_max = round(max(age), 1),
        ty_le_nam = round(mean(gender_numeric) * 100, 1)
      )
    
    # In thống kê
    print("Thống kê các nhóm:")
    print(group_stats)
    
  })
}

# 10. Ngắt kết nối Spark -----------------------------------------------
spark_disconnect(sc)