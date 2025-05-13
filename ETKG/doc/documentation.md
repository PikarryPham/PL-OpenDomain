# Solution Document for Building an Educational Temporal KG (ETKG)

## 1. Solution Overview

This solution builds an Educational Temporal KG (ETKG) using Neo4j to store and manage data from various sources. The TKG is designed to represent relationships between users, questions, options, learning history, and related concepts, with associated temporal information to track changes over time.

## 2. Data Model

### 2.1 Node Types

#### User Node
- Label: `User`
- Key properties: `user_id`, `username`, `email`, `created_time_original`, `updated_time`

#### Question Node
- Label: `Question`
- Key properties: `id`, `question_text`, `type`, `form`, `created_at`

#### Option Node
- Label: `Option`
- Key properties: `id`, `option_text`, `question_id`

#### Entry Node
- Label: `Entry`
- Key properties: `entry_id`, `url`, `title`, `timestamp`, `pageview_count`, `window_time_details`, `user_id`

#### Concept Nodes
1. **Topic Node** (Level 1 - root)
   - Label: `Topic`
   - Key properties: `uri`, `label`

2. **Category Node** (Level 2 - parent)
   - Label: `Category`
   - Key properties: `uri`, `label`

3. **RelatedConcept Node** (Level 3 - child)
   - Label: `RelatedConcept`
   - Key properties: `uri`, `label`, `language`, `relationshipType`, `abstract`, `comment`

### 2.2 Relationship Types

#### User-Entry Relationship
- Type: `HAS`
- Properties: `timestamp`

#### User-Option Relationship
- Type: `CHOOSE`
- Properties: `timestamp`, `order`

#### Question-Option Relationship
- Type: `HAS`
- Properties: `timestamp`

#### Entry-Topic Relationship
- Type: `RELATE_TO`
- Properties: `weightDurationPerSession`, `weightView`, `timestamp`

#### Category-Topic Relationship
- Type: `HAS_ROOT`
- Properties: `timestamp`

#### RelatedConcept-Category Relationship
- Type: `HAS_PARENT`
- Properties: `timestamp`

### 2.3 Graph Model

```
(User) -[CHOOSE]-> (Option) <-[HAS]- (Question)
   |
   +-[HAS]-> (Entry) -[RELATE_TO]-> (Topic) <-[HAS_ROOT]- (Category) <-[HAS_PARENT]- (RelatedConcept)
```

## 3. Source Code Structure

### 3.1 Directory Structure

```
project/
├── main.py                  # Entry point
├── main_test.py             # Entry point with limited data for testing
├── config.py                # Configuration
├── db_connector.py          # Connection to Neo4j
├── node_manager.py          # Node management
├── relationship_manager.py  # Relationship management
├── validate_kg.py           # KG validation and verification
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # Reading data from input files
│   └── data_processor.py    # Processing and normalizing data
└── models/
    ├── __init__.py
    ├── user.py              # Model for User
    ├── question.py          # Model for Question
    ├── option.py            # Model for Option
    ├── entry.py             # Model for Entry
    └── concept.py           # Model for Concept
```

### 3.2 Main Modules

#### 3.2.1 Neo4j Connector Module (db_connector.py)

This module provides functionalities to connect and interact with Neo4j, including:
- Connecting to Neo4j
- Executing Cypher queries
- Creating constraints and indexes
- Deleting data

#### 3.2.2 Node Management Module (node_manager.py)

This module provides functionalities to create and update nodes in Neo4j, including:
- Creating or updating User nodes
- Creating or updating Question nodes
- Creating or updating Option nodes
- Creating or updating Entry nodes
- Creating or updating Concept nodes (Topic, Category, RelatedConcept)

#### 3.2.3 Relationship Management Module (relationship_manager.py)

This module provides functionalities to create and update relationships between nodes in Neo4j, including:
- Creating or updating relationships between User and Entry
- Creating or updating relationships between User and Option
- Creating or updating relationships between Question and Option
- Creating or updating relationships between Entry and Topic
- Creating or updating relationships between Category and Topic
- Creating or updating relationships between RelatedConcept and Category

#### 3.2.4 Data Loader Module (utils/data_loader.py)

This module provides functionalities to read data from JSON files, including:
- Reading data from JSON files
- Reading user data
- Reading question data
- Reading option data
- Reading learning history data
- Reading sample output data

#### 3.2.5 Data Processor Module (utils/data_processor.py)

This module provides functionalities to process and normalize data, including:
- Processing user data
- Processing question data
- Processing option data
- Processing entry data
- Processing concept data
- Searching data

#### 3.2.6 Model Module (models/)

This module provides object classes to represent the data, including:
- User model
- Question model
- Option model
- Entry model
- Concept model

## 5. Usage Guide

### 5.1 System Requirements

- Python 3.6+
- Neo4j 4.0+
- Python libraries: neo4j, py2neo

### 5.2 Installation

1. Install Neo4j:
   ```bash
   # Ubuntu
   wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
   echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
   sudo apt-get update
   sudo apt-get install -y neo4j
   ```

2. Install Python libraries:
   ```bash
   pip install neo4j py2neo
   ```

3. Configure Neo4j:
   - Start Neo4j: `sudo systemctl start neo4j`
   - Change the default password: `cypher-shell -u neo4j -p neo4j "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'password123'"`

### 5.3 Running the Application

1. Configure Neo4j connection info in `config.py`:
   ```python
   NEO4J_URI = "bolt://localhost:7687"
   NEO4J_USER = "neo4j"
   NEO4J_PASSWORD = "password123"
   ```

2. Run the application:
   ```bash
   python main.py
   ```

3. Verify results:
   ```bash
   python validate_kg.py
   ```

### 5.4 Querying the TKG

After building the TKG, you can query the data using Cypher queries in the Neo4j Browser:

1. Open Neo4j Browser: `http://localhost:7474`
2. Log in with the configured credentials
3. Execute Cypher queries


## 6. Conclusion

This solution successfully builds an ETKG in Neo4j to represent relationships between users, questions, options, learning history, and related concepts, with associated temporal information. This TKG can be used for user behavior analysis, discovering relationships between concepts, and many other applications.