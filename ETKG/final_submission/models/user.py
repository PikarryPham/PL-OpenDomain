import sys
sys.path.append('/home/ubuntu')
from config import *

class User:
    """
    Model cho User Node
    """
    def __init__(self, user_data):
        self.user_id = user_data.get('user_id')
        self.username = user_data.get('username')
        self.email = user_data.get('email')
        self.created_time_original = user_data.get('created_time_original')
        self.updated_time = user_data.get('updated_time')
        self.preferred_areas = user_data.get('preferred_areas', [])
        self.preferred_content_types = user_data.get('preferred_content_types', [])
        self.preferred_learn_style = user_data.get('preferred_learn_style', [])
        self.education_lv = user_data.get('education_lv', [])
    
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary
        
        Returns:
            dict: Dictionary chứa thông tin của User
        """
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'created_time_original': self.created_time_original,
            'updated_time': self.updated_time
        }
    
    def get_cypher_properties(self):
        """
        Lấy các thuộc tính cho Cypher query
        
        Returns:
            dict: Dictionary chứa các thuộc tính cho Cypher query
        """
        properties = self.to_dict()
        # Loại bỏ các giá trị None
        return {k: v for k, v in properties.items() if v is not None}
    
    def get_preferred_options(self):
        """
        Lấy danh sách các lựa chọn ưa thích của người dùng
        
        Returns:
            list: Danh sách các lựa chọn ưa thích
        """
        preferred_options = []
        
        # Thêm preferred_areas
        for area in self.preferred_areas:
            preferred_options.append({
                'form': 'preferred_areas',
                'option': area.get('option'),
                'order': area.get('order')
            })
        
        # Thêm preferred_content_types
        for content_type in self.preferred_content_types:
            preferred_options.append({
                'form': 'preferred_content_types',
                'option': content_type.get('option'),
                'order': content_type.get('order')
            })
        
        # Thêm preferred_learn_style
        for learn_style in self.preferred_learn_style:
            preferred_options.append({
                'form': 'preferred_learn_style',
                'option': learn_style.get('option'),
                'order': learn_style.get('order')
            })
        
        # Thêm education_lv
        for education in self.education_lv:
            preferred_options.append({
                'form': 'education_lv',
                'option': education.get('option'),
                'order': education.get('order')
            })
        
        return preferred_options
