import sys
sys.path.append('/home/ubuntu')
from config import *

class Option:
    """
    Model cho Option Node
    """
    def __init__(self, option_data):
        self.id = option_data.get('id')
        self.question_id = option_data.get('question_id')
        self.option_text = option_data.get('option_text')
    
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary
        
        Returns:
            dict: Dictionary chứa thông tin của Option
        """
        return {
            'id': self.id,
            'question_id': self.question_id,
            'option_text': self.option_text
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
