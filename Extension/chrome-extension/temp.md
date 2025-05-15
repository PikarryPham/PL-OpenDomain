%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#f8f8f8', 'lineColor': '#555', 'textColor': '#333', 'fontFamily': 'Virgil', 'fontSize': '14px' } } }%%
graph LR
    %% Define Styles %%
    style User fill:#FFDAB9,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style AirflowOrchestrator fill:#ADD8E6,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style CrawlTask fill:#ADD8E6,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style CheckCrawled fill:#ADD8E6,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5,shape:diamond
    style ExtractData fill:#ADD8E6,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style MonitorControl fill:#ADD8E6,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5,shape:diamond
    style CookieProxy fill:#E6E6FA,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style YouTubeAPI fill:#E6E6FA,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style MinIO fill:#D3D3D3,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style PostgreSQL fill:#D3D3D3,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style EndSession fill:#F08080,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5

    %% Define Nodes and Edges %%
    User["User (Client)"] -- "Keywords (Channel/Playlist/Topic)" --> AirflowOrchestrator((Apache Airflow));
    AirflowOrchestrator -- "Receive Keywords" --> CheckCrawled{"Already Crawled?"};

    %% Decision Path: Already Crawled? %%
    CheckCrawled -- "YES --> Next Item" --> AirflowOrchestrator; %% Dashed line for optional/alternative path %%
    linkStyle 1 stroke-dasharray: 5 5;

    CheckCrawled -- "NO --> Initiate Crawl" --> CrawlTask["Crawl Task"];

    %% Crawling Path %%
    CrawlTask -- "Request Data" --> CookieProxy["Cookie & Proxy Manager\n(Rotate Headers)"];
    CookieProxy -- "Managed Request" --> YouTubeAPI["Remote YouTube API"];
    YouTubeAPI -- "Raw Data (Video/Channel/Playlist)" --> CrawlTask;

    %% Data Processing and Storage Path %%
    CrawlTask -- "Data to Process" --> ExtractData["Extract & Process Data\n(Audio, Transcript, Metadata)"];
    ExtractData -- "Object Data (.wav, .txt)" --> MinIO["MinIO Object Storage"];
    ExtractData -- "Metadata (Video, Channel, Subs)" --> PostgreSQL["PostgreSQL DB"];

    %% Monitoring and Control Path %%
    MinIO -- "Upload Status" --> MonitorControl{"Complete / Error Rate < 10%?"};
    PostgreSQL -- "Upload Status" --> MonitorControl;
    CrawlTask -- "Task Status" --> MonitorControl;

    %% Decision Path: End Session? %%
    MonitorControl -- "YES" --> EndSession[("End Session")];
    MonitorControl -- "NO (Continue/Retry)" ---> AirflowOrchestrator;

    %% Define Subgraphs for Clarity %%
    subgraph "Orchestration (Pastel Blue)"
        AirflowOrchestrator
        CheckCrawled
        CrawlTask
        ExtractData
        MonitorControl
    end

    subgraph "External Services (Pastel Purple)"
        CookieProxy
        YouTubeAPI
    end

    subgraph "Data Storage (Grey)"
        MinIO
        PostgreSQL
    end

    subgraph "Initiation (Pastel Orange)"
        User
    end

     subgraph "Termination (Red)"
        EndSession
    end
