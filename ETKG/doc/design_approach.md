# Design of Data Processing Method for Temporal Knowledge Graph

## 1. Neo4j Data Model

### 1.1 Node Types

#### User Node
- Label: `User`
- Properties:
  - `user_id`: ID of the user (primary key)
  - `username`: Username
  - `email`: User's email
  - `created_time_original`: Original creation time
  - `updated_time`: Update time
  - Other properties from users_data_sample.json

#### Question Node
- Label: `Question`
- Properties:
  - `id`: ID of the question (primary key)
  - `question_text`: Question content
  - `type`: Question type (single/multiple)
  - `form`: Form type (preferred_learn_style, preferred_content_types, etc.)
  - `created_at`: Creation time

#### Option Node
- Label: `Option`
- Properties:
  - `id`: ID of the option (primary key)
  - `option_text`: Option content
  - `question_id`: Related question ID

#### Entry Node
- Label: `Entry`
- Properties:
  - `entry_id`: Entry ID (primary key)
  - `url`: Page URL
  - `title`: Page title
  - `timestamp`: Access time
  - `pageview_count`: Page view count
  - `capped_time_on_page`: Time spent on page
  - Other properties from history_learning_data_sample.json

#### Concept Node
- Divided into 3 types based on level:
  1. **Topic Node** (Level 1 - root)
     - Label: `Topic`
     - Properties:
       - `uri`: Topic URI
       - `label`: Topic label
  
  2. **Category Node** (Level 2 - parent)
     - Label: `Category`
     - Properties:
       - `uri`: Category URI
       - `label`: Category label
  
  3. **RelatedConcept Node** (Level 3 - child)
     - Label: `RelatedConcept`
     - Properties:
       - `uri`: Concept URI
       - `label`: Concept label
       - `language`: Language
       - `relationshipType`: Relationship type
       - `abstract`: Abstract
       - `comment`: Comment

### 1.2 Relationship Types

#### User-Entry Relationship
- Type: `HAS`
- Properties:
  - `timestamp`: Time of the entry

#### User-Option Relationship
- Type: `CHOOSE`
- Properties:
  - `timestamp`: User update time (`updated_time`)
  - `order`: Choice order

#### Question-Option Relationship
- Type: `HAS`
- Properties:
  - `timestamp`: Question creation time (`created_at`)

#### Entry-Topic Relationship
- Type: `RELATE_TO`
- Properties:
  - `weightDurationPerSession`: Time spent on page (`capped_time_on_page`)
  - `weightView`: Page view count (`pageview_count`)
  - `timestamp`: Current time when creating the relationship

#### Category-Topic Relationship
- Type: `HAS_ROOT`
- Properties:
  - `timestamp`: Current time when creating the relationship

#### RelatedConcept-Category Relationship
- Type: `HAS_PARENT`
- Properties:
  - `timestamp`: Current time when creating the relationship

## 2. Data Processing Workflow

### 2.1 Neo4j Installation and Configuration
1. Install Neo4j (using Docker or direct installation)
2. Configure Neo4j to accept connections from the Python application
3. Create constraints and indexes to optimize performance

### 2.2 Reading and Processing Data
1. Read data from JSON files
2. Convert data into Python objects
3. Process and normalize data

### 2.3 Creating and Updating Nodes/Relationships in Neo4j
1. Check for the existence of a node before creating:
   - If the node does not exist: create the node
   - If the node exists: update the node's properties
2. Check for the existence of a relationship before creating:
   - If the relationship does not exist: create the relationship
   - If the relationship exists: update the relationship's properties

### 2.4 Detailed Processing Steps
1. **Process User Node**:
   - Read data from users_data_sample.json
   - Create or update the User Node

2. **Process Question and Option Nodes**:
   - Read data from question_data_sample.json and options_data_sample.json
   - Create or update the Question and Option Nodes
   - Create `HAS` relationships between Question and Option

3. **Process User-Option Relationship**:
   - Based on `preferred_*` information in users_data_sample.json
   - Create `CHOOSE` relationships between User and Option

4. **Process Entry Node**:
   - Read data from history_learning_data_sample.json
   - Create or update the Entry Node
   - Create `HAS` relationships between User and Entry

5. **Process Concept Nodes and Relationships**:
   - Read data from final_sample_output.json
   - Create or update Topic, Category, and RelatedConcept Nodes
   - Create `RELATE_TO` relationships between Entry and Topic
   - Create `HAS_ROOT` relationships between Category and Topic
   - Create `HAS_PARENT` relationships between RelatedConcept and Category

## 3. Deduplication Strategy

### 3.1 Deduplicate Nodes
- Use `MERGE` instead of `CREATE` in Cypher queries
- Check for node existence based on primary key (ID)
- For Concept Nodes, check based on URI

### 3.2 Deduplicate Relationships
- Check for relationship existence before creating
- Use `MERGE` instead of `CREATE` in Cypher queries
- Update relationship properties if it already exists

## 4. Codebase Structure

### 4.1 Directory Structure
```
project/
├── main.py                  # Entry point
├── config.py                # Configuration
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # Read data from input files
│   └── data_processor.py    # Process and normalize data
├── models/
│   ├── __init__.py
│   ├── user.py              # Model for User
│   ├── question.py          # Model for Question
│   ├── option.py            # Model for Option
│   ├── entry.py             # Model for Entry
│   └── concept.py           # Model for Concept
└── neo4j/
    ├── __init__.py
    ├── connector.py         # Connect to Neo4j
    ├── node_manager.py      # Manage nodes
    └── relationship_manager.py  # Manage relationships
```

### 4.2 Main Modules
1. **data_loader.py**: Read input data
2. **data_processor.py**: Process and normalize data
3. **connector.py**: Connect to Neo4j
4. **node_manager.py**: Manage nodes (create, update, delete)
5. **relationship_manager.py**: Manage relationships (create, update, delete)
6. **main.py**: Entry point, orchestrates the processing

## 5. Data Processing Techniques

### 5.1 Error Handling
- Catch and handle exceptions
- Log for monitoring and debugging

### 5.2 Performance Optimization
- Create indexes for frequently queried properties
- Use parameterized queries to prevent SQL injection and improve performance