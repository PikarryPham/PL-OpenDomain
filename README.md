# Enhancing Personalized Learning With Real-Time Data Warehousing and Linked Open Data Integration For Open Knowledge Domain Documentation

## Overview

This document provides a comprehensive guide to the implementation of the proposed method described in the research paper. The project is structured into multiple stages, each with corresponding code repositories and documentation.

## Note about data request
Please note that the data files contained in the folders ```/extension/chrome-extension/chatbot/backend/src/data/``` and ```/ETKG/upload/``` are only a partial data extracted from the actual data. Because the data contain private information of learners so it is only available upon request. Please dm me via email for request data: trang.pham@jaist.ac.jp. 

## Project Structure

The implementation is organized according to the stages of the proposed method:

1. **Stage 1 & 3.1: Learner's Learning Data Collection & Learning Data With Knowledge From LOD Integration**
   - Stage 1 Code location: Folder 
   ```
      /extension/chrome-extension/Chrome_Plugin_Base
      /extension/chrome-extension/API
   ```
   - Documentation: `/extension/chrome-extension/doc/`
   - Stage 3.1 Code location: `/extension/chrome-extension/chatbot` folder
   - Documentation: `/extension/chrome-extension/chatbot/doc/`

2. **Stage 2: ETL Processing On Data Lakehouse**
   - Code location: `/dwh-lab` folder
   - Documentation: `/dwh-lab/doc/`

3. **Stage 3.2: ETKG Construction**
   - Code location: `/ETKG` folder
   - Documentation: `/ETKG/doc/`

## Implementation Priority

It is strongly recommended to follow the implementation order as presented in the research paper. Each stage builds upon the previous ones, and the correct sequence ensures proper data flow and integration.