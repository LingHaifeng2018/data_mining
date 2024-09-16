# data_preprocessing/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from scipy import stats

# 数据类型转换类
# 数据类型转换类
class DataTypeConverter:
    @staticmethod
    def convert(data, continuous_column_names=None, category_column_names=None):
        if data is None:
            raise ValueError("传入的数据是None")

        # 处理连续变量列中的非数值字符串
        if continuous_column_names:
            for column_name in continuous_column_names:
                if column_name in data.columns:
                    # 尝试转换为数值，非数值的部分将被设置为 NaN
                    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
                    # 填充 NaN 值为列的均值
                    mean_value = data[column_name].mean()
                    data[column_name] = data[column_name].fillna(mean_value)
                    # 确保数据类型为浮点数
                    data[column_name] = data[column_name].astype(float)

        # 处理分类变量列中的 NaN 值
        if category_column_names:
            for category_column_name in category_column_names:
                if category_column_name in data.columns:
                    # 替换分类变量列中的 NaN 值为列的众数
                    mode_value = data[category_column_name].mode()[0]
                    data[category_column_name] = data[category_column_name].fillna(mode_value)

                    # 确保数据类型为整型
                    data[category_column_name] = data[category_column_name].astype(int)

        return data
# 缺失值处理类
# 缺失值处理类
class MissingValuesHandler:
    @staticmethod
    def handle(data, fill_strategy='mean'):
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                # 对于数值类型数据，使用指定的填充策略
                if fill_strategy == 'mean':
                    mean_value = data[column].mean(skipna=True)
                    data[column] = data[column].fillna(mean_value)
                elif fill_strategy == 'median':
                    median_value = data[column].median(skipna=True)
                    data[column] = data[column].fillna(mean_value)
                else:
                    raise ValueError("请为数值型数据选择填充策略: 'mean' 或 'median'。")
            else:
                # 对于非数值类型数据，使用众数填充
                mode_value = data[column].mode()[0]
                data[column] = data[column].fillna(mode_value)
        return data


# 异常值处理类
# 异常值处理类
class OutlierHandler:
    @staticmethod
    def handle(data, method='z_score', threshold=3):
        if method == 'z_score':
            z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
            data = data[(z_scores < threshold).all(axis=1)]
        else:
            raise ValueError("不支持的异常值处理方法")

        return data
# 数据编码类
# 数据编码类
class DataEncoder:
    @staticmethod
    def encode(data, category_column_names=None):
        if data is None:
            raise ValueError("传入的数据是None")

        if category_column_names:
            for category_column_name in category_column_names:
                if category_column_name in data.columns:
                    one_hot_encoder = OneHotEncoder()
                    encoded = one_hot_encoder.fit_transform(data[[category_column_name]])
                    feature_names = one_hot_encoder.get_feature_names_out([category_column_name])
                    encoded_df = pd.DataFrame(encoded.toarray(), columns=feature_names, index=data.index)
                    data = pd.concat([data.drop(columns=[category_column_name]), encoded_df], axis=1)
                else:
                    raise ValueError(f"数据中缺少 '{category_column_name}' 列")
        else:
            raise ValueError("未指定要编码的分类列")

        return data
# 数据标准化和归一化类
class DataScaler:
    @staticmethod
    def scale(data, method='standard', exclude_columns=None):
        numeric_data = data.select_dtypes(include=[np.number])
        if exclude_columns:
            numeric_data = numeric_data.drop(columns=exclude_columns)
        if method == 'standard':
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(scaler.fit_transform(numeric_data), columns=numeric_data.columns, index=data.index)
            return data_scaled
        elif method == 'min_max':
            min_max_scaler = MinMaxScaler()
            data_normalized = pd.DataFrame(min_max_scaler.fit_transform(numeric_data), columns=numeric_data.columns, index=data.index)
            return data_normalized
        else:
            raise ValueError("不支持的缩放方法")
# 数据预处理管理类
# 数据预处理管理类
class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self, continuous_column_names=None, category_column_names=None, fill_strategy='mean', method='z_score', threshold=3, encoding_columns=None, scaling_method='standard', exclude_columns=None):
        self.data = DataTypeConverter.convert(self.data, continuous_column_names=continuous_column_names, category_column_names=category_column_names)
        self.data = MissingValuesHandler.handle(self.data, fill_strategy=fill_strategy)
        self.data = OutlierHandler.handle(self.data, method=method, threshold=threshold)
        self.data = DataEncoder.encode(self.data, category_column_names=encoding_columns)
        if scaling_method == 'standard':
            self.data_scaled = DataScaler.scale(self.data, method='standard', exclude_columns=exclude_columns)
        elif scaling_method == 'min_max':
            self.data_normalized = DataScaler.scale(self.data, method='min_max', exclude_columns=exclude_columns)
        else:
            raise ValueError("不支持的缩放方法")
        return self.data, self.data_scaled, self.data_normalized