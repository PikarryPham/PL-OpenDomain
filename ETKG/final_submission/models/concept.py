import sys
sys.path.append('/home/ubuntu')
from config import *

class Concept:
    """
    Model cho Concept Node (Topic, Category, RelatedConcept)
    """
    def __init__(self, concept_data, concept_type):
        """
        Khởi tạo đối tượng Concept
        
        Args:
            concept_data (dict): Dữ liệu concept
            concept_type (str): Loại concept (Topic, Category, RelatedConcept)
        """
        self.uri = concept_data.get('uri')
        self.label = concept_data.get('label')
        self.concept_type = concept_type
        
        # Các thuộc tính bổ sung cho RelatedConcept
        if concept_type == 'RelatedConcept':
            self.language = concept_data.get('language')
            self.relationship_type = concept_data.get('relationshipType')
            self.abstract = concept_data.get('abstract')
            self.comment = concept_data.get('comment')
    
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary
        
        Returns:
            dict: Dictionary chứa thông tin của Concept
        """
        result = {
            'uri': self.uri,
            'label': self.label,
            'concept_type': self.concept_type
        }
        
        # Thêm các thuộc tính bổ sung cho RelatedConcept
        if self.concept_type == 'RelatedConcept':
            result.update({
                'language': self.language,
                'relationship_type': self.relationship_type,
                'abstract': self.abstract,
                'comment': self.comment
            })
        
        return result
    
    def get_cypher_properties(self):
        """
        Lấy các thuộc tính cho Cypher query
        
        Returns:
            dict: Dictionary chứa các thuộc tính cho Cypher query
        """
        properties = self.to_dict()
        # Loại bỏ các giá trị None
        return {k: v for k, v in properties.items() if v is not None}
    
    @classmethod
    def create_topic(cls, topic_data):
        """
        Tạo đối tượng Topic (Level 1)
        
        Args:
            topic_data (dict): Dữ liệu topic
            
        Returns:
            Concept: Đối tượng Topic
        """
        return cls(topic_data, 'Topic')
    
    @classmethod
    def create_category(cls, category_data):
        """
        Tạo đối tượng Category (Level 2)
        
        Args:
            category_data (dict): Dữ liệu category
            
        Returns:
            Concept: Đối tượng Category
        """
        return cls(category_data, 'Category')
    
    @classmethod
    def create_related_concept(cls, related_concept_data):
        """
        Tạo đối tượng RelatedConcept (Level 3)
        
        Args:
            related_concept_data (dict): Dữ liệu related concept
            
        Returns:
            Concept: Đối tượng RelatedConcept
        """
        return cls(related_concept_data, 'RelatedConcept')
