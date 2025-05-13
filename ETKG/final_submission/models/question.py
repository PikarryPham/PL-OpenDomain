import sys
sys.path.append('/home/ubuntu')
from config import *

class Question:
    """
    Model cho Question Node
    """
    def __init__(self, question_data):
        self.id = question_data.get('id')
        self.question_text = question_data.get('question_text')
        self.type = question_data.get('type')
        self.form = question_data.get('form')
        self.created_at = question_data.get('created_at')
    
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary
        
        Returns:
            dict: Dictionary chứa thông tin của Question
        """
        return {
            'id': self.id,
            'question_text': self.question_text,
            'type': self.type,
            'form': self.form,
            'created_at': self.created_at
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
