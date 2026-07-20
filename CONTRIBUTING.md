# Contributing to Wafer Inspection

Thank you for your interest in contributing to the Wafer Inspection project! 🎉

## 贡献指南

### 如何贡献 / How to Contribute

**🇨🇳 中文：**
1. **Fork** 本仓库
2. 创建你的特性分支：`git checkout -b feat/your-feature`
3. 提交你的改动：`git commit -m "feat: add awesome feature"`
4. 推送到你的 Fork：`git push origin feat/your-feature`
5. 创建 Pull Request

**🇬🇧 English:**
1. **Fork** this repository
2. Create your feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add awesome feature"`
4. Push to your fork: `git push origin feat/your-feature`
5. Open a Pull Request

---

### Development Environment

#### wafer-inspection (PyTorch)
```bash
cd wafer-inspection
pip install -e .
# Run tests
pytest tests/
```

#### wafer-backend (Spring Boot)
```bash
cd wafer-backend
./mvnw spring-boot:run
```

#### wafer-showcase (React)
```bash
cd wafer-showcase
npm install
npm run dev
```

#### wafer-lora-service (FastAPI)
```bash
cd wafer-lora-service
pip install -r requirements.txt
python main.py
```

#### wafer-qwen-service (FastAPI)
```bash
cd wafer-qwen-service
pip install -r requirements.txt
# Requires llama.cpp server with Qwen2.5-VL GGUF model
python -m uvicorn main:app --port 8001
```

### Full Stack (Docker)
```bash
docker-compose up -d
```

---

### Code Style

- **Python**: Follow PEP 8, use type hints
- **Java**: Follow Spring Boot conventions, use Lombok
- **JavaScript/React**: Use ESLint + Prettier config in the project
- **Commit Messages**: Use conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)
- **Documentation**: Keep bilingual (EN/ZH) for public-facing docs

### Testing

- All new features should include tests
- Python: `pytest`
- Java: `mvn test`
- Run existing tests before submitting PR

### Model Files

**Do NOT commit model weights** (`.pt`, `.gguf`, `.safetensors`, etc.) to git.
- Small reference weights in `reference/` are exceptions
- Use Hugging Face or Git LFS for large models

---

## Pull Request Process

1. Ensure tests pass and code builds cleanly
2. Update documentation if needed
3. The PR title should use conventional commit format
4. At least one maintainer review required before merge

## Code of Conduct

All contributors must adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

**Questions?** Open an issue or start a discussion. We're here to help!
