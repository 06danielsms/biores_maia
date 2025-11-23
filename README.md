# BIORES MAIA - Medical AI Assistant

A full-stack application for biomedical text preprocessing, analysis, and simplification.

## 🚀 Quick Start

Run the setup script:
```bash
./quick_start.sh
```

Or follow the manual instructions in [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).

## 📁 Project Structure

```
biores_maia/
├── biores_api/          # FastAPI backend (Hexagonal Architecture)
│   ├── domain/          # Domain models and ports (interfaces)
│   ├── application/     # Use cases (business logic)
│   ├── infrastructure/  # Adapters (implementations)
│   └── api/            # REST API layer
├── biores_ui/          # React + TypeScript frontend
│   ├── src/
│   │   ├── services/   # API client
│   │   ├── pages/      # UI pages
│   │   └── components/ # Reusable components
│   └── .env            # Environment configuration
├── data/               # Datasets (DVC tracked)
├── scripts/            # Utility scripts
└── jupyter/            # Notebooks for experiments
```

## 🎯 Features

### Backend (FastAPI)
- ✅ **Hexagonal Architecture** (Ports & Adapters pattern)
- ✅ **9 REST Endpoints** for text preprocessing, metrics, and plotting
- ✅ **Text Cleaning**: HTML stripping, PHI de-identification, normalization
- ✅ **Text Chunking**: spaCy-based sentence splitting
- ✅ **Readability Metrics**: Flesch, FK Grade, Gunning Fog, SMOG, etc.
- ✅ **Data Visualization**: Distribution plots for analysis
- ✅ **Dependency Injection** with FastAPI Depends
- ✅ **Automatic API Documentation** (OpenAPI/Swagger)

### Frontend (React + TypeScript)
- ✅ **Type-Safe API Client** with full TypeScript support
- ✅ **Drag & Drop File Upload** with validation
- ✅ **Multiple File Selection** for batch processing
- ✅ **Batch Processing Mode** - Process multiple files sequentially
- ✅ **Real-Time Progress Indicator** (Processing 3/10...)
- ✅ **Real-Time Preprocessing** with configurable options
- ✅ **Text Comparison** (Original vs Processed)
- ✅ **Batch Results Table** with metrics for each file
- ✅ **Metrics Display** with visual cards
- ✅ **Error Handling** with user-friendly messages
- ✅ **Loading States** for better UX
- ✅ **Responsive Design** with Tailwind CSS

## 🏃 Running the Application

### Backend Server
```bash
cd biores_api
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Frontend Server
```bash
cd biores_ui
npm install
npm run dev
```

Access at: http://localhost:5173

## 🧪 Testing

Run the integration test:
```bash
python test_api_integration.py
```

Test with sample data:
- **Single file**: Upload `sample_clinical_trial.txt` in the UI
- **Batch processing**: Select multiple files from `test_files/` directory
- Configure preprocessing options
- View processed results and metrics
- Check batch results table for multiple files

### Batch Processing Test
```bash
# Navigate to test files directory
cd test_files/

# You'll find sample files:
# - sample_1_diabetes.txt
# - sample_2_hypertension.txt
# - sample_3_melanoma.txt
# - sample_4_alzheimers.txt

# Select all files in the UI to test batch mode
```

## 📚 Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Complete setup guide
- **[BATCH_PROCESSING_UPDATE.md](BATCH_PROCESSING_UPDATE.md)** - Batch processing features
- **[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)** - Implementation details
- **[CHEAT_SHEET.md](CHEAT_SHEET.md)** - Quick command reference
- **[biores_api/ARCHITECTURE.md](biores_api/ARCHITECTURE.md)** - Backend architecture
- **[biores_api/PROJECT_SUMMARY.md](biores_api/PROJECT_SUMMARY.md)** - API documentation

## 🔧 Configuration

### Backend Dependencies
```bash
pip install -e .
pip install beautifulsoup4 lxml ftfy unidecode
python -m spacy download en_core_web_sm
```

### Frontend Environment
Create `biores_ui/.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```

## 🛠️ Technologies

### Backend
- **FastAPI**: Modern Python web framework
- **spaCy**: NLP for text processing
- **Textstat**: Readability metrics
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting
- **Pydantic**: Data validation

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Fetch API**: HTTP client

## 📊 API Endpoints

### Cleaning
- `POST /api/v1/cleaning/clean-text` - Clean single text
- `POST /api/v1/cleaning/clean-batch` - Clean multiple files

### Metrics
- `POST /api/v1/metrics/compute-single` - Compute metrics for text
- `POST /api/v1/metrics/compute-batch` - Compute for multiple files

### Plotting
- `POST /api/v1/plotting/feature-distributions` - Plot feature distributions
- `POST /api/v1/plotting/metrics-comparison` - Compare metrics
- `POST /api/v1/plotting/correlation-matrix` - Plot correlations
- `POST /api/v1/plotting/boxplot-comparison` - Boxplot comparisons

### Health
- `GET /api/v1/health` - Health check

## 🎓 Architecture

The backend follows **Hexagonal Architecture** (Ports & Adapters):

```
API Layer (FastAPI Routes)
        ↓
Application Layer (Use Cases)
        ↓
Domain Layer (Ports - Interfaces)
        ↓
Infrastructure Layer (Adapters - Implementations)
```

This ensures:
- **Testability**: Easy to mock dependencies
- **Flexibility**: Swap implementations without changing business logic
- **Maintainability**: Clear separation of concerns
- **Scalability**: Easy to add new features

## 🔒 Security

- CORS enabled for local development
- File type validation
- PHI de-identification included
- Input sanitization
- Error messages don't expose sensitive data

## 📈 Next Steps

- [ ] Add batch file processing
- [ ] Export processed texts
- [ ] Visualization dashboard
- [ ] User authentication
- [ ] Session history
- [ ] Custom stop words
- [ ] Advanced chunking controls

## 🤝 Contributing

This project follows clean architecture principles. When adding features:

1. Define interfaces in `domain/ports/`
2. Implement business logic in `application/use_cases/`
3. Create adapters in `infrastructure/adapters/`
4. Expose via routes in `api/routes/`
5. Add frontend integration in `biores_ui/src/`

## 📝 Version History

- **v1.0.0** - Initial release with preprocessing endpoints
- **v1.1.0** - Frontend integration complete
- **v1.2.0** - Batch processing & drag-drop upload added

## Herramientas Instaladas

- **Python**: 3.12.3
- **Git**: 2.43.0
- **DVC**: 3.63.0

## Conexión SSH

### Información del Servidor
- **IP**: 52.91.22.48
- **Usuario**: ubuntu
- **Puerto**: 22
- **KEY**: BIORES.pem
