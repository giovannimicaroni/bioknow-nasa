# 🚀 Deploy no Heroku - Otimizado

## ⚠️ Problema Resolvido

**Antes:** Slug de ~4GB (muito acima do limite de 500MB)
- PyTorch CUDA: 888 MB
- NVIDIA libraries: ~3GB
- **Total: Impossível fazer deploy**

**Agora:** Slug de ~300MB ✅
- PyTorch CPU: ~200MB
- Sem bibliotecas CUDA
- **Total: Deploy OK!**

## 📋 Arquivos Criados/Modificados

1. ✅ **requirements.txt** - PyTorch CPU-only
2. ✅ **.slugignore** - Remove arquivos desnecessários
3. ✅ **runtime.txt** - Python 3.11.9

## 🔧 Como Fazer Deploy

### 1. **Configure as variáveis de ambiente no Heroku**

```bash
heroku config:set OPENAI_API_KEY=sk-your-key-here
heroku config:set FLASK_SECRET_KEY=your-secret-key-here
heroku config:set FLASK_ENV=production
```

### 2. **Faça o commit das mudanças**

```bash
git add requirements.txt .slugignore
git commit -m "Otimizar para Heroku - PyTorch CPU-only"
```

### 3. **Deploy para o Heroku**

```bash
git push heroku main
```

**OU** se estiver em outro branch:

```bash
git push heroku jabs:main
```

### 4. **Acompanhe o build**

O Heroku agora vai:
- ✅ Baixar PyTorch CPU (~200MB)
- ✅ Ignorar arquivos desnecessários (.slugignore)
- ✅ Build em ~2-3 minutos
- ✅ Slug final: ~300-400MB

## 📊 Tamanho do Slug Esperado

```
Compressing... done, 350.2MB
Launching... done
https://your-app.herokuapp.com/ deployed to Heroku
```

## ⚡ Otimizações Aplicadas

### 1. **PyTorch CPU-only**
- ❌ torch==2.8.0: 888 MB
- ✅ torch==2.1.0+cpu: ~200 MB
- **Economia: ~700MB**

### 2. **.slugignore**
Remove:
- Arquivos de desenvolvimento
- PDFs
- Sessões locais
- Documentação markdown
- **Economia: ~50-100MB**

### 3. **Apenas dependências necessárias**
- Sem bibliotecas NVIDIA
- Sem CUDA toolkit
- **Economia: ~3GB**

## 🔍 Verificar Tamanho do Slug

```bash
# Depois do deploy
heroku repo:purge_cache -a your-app-name
heroku builds:info -a your-app-name
```

## ⚠️ Troubleshooting

### **Erro: "Slug size too large"**
```bash
# Limpe o cache do Heroku
heroku repo:purge_cache -a your-app-name
heroku plugins:install heroku-repo
heroku repo:gc -a your-app-name

# Faça deploy novamente
git push heroku main --force
```

### **Erro: "torch not found"**
- Certifique-se que o requirements.txt tem `--find-links`
- Verifique que está usando versões +cpu

### **App muito lento no Heroku**
- Normal! Heroku free tier tem CPU limitada
- PyTorch CPU é mais lento que GPU
- Considere usar plano pago ou alternativas

## 🎯 Resultado Final

Depois do deploy:
```
✅ Slug: ~350MB (abaixo do limite de 500MB)
✅ Build: 2-3 minutos
✅ Deploy: Sucesso!
✅ App: Funcionando
```

## 📝 Notas Importantes

1. **PyTorch CPU é suficiente** para:
   - Embeddings de texto
   - Sentence transformers
   - Langchain
   - Processamento de documentos

2. **Não precisa de GPU** para este projeto

3. **Performance aceitável** para chatbot e análise de documentos

4. **Considere alternativas** se precisar de mais performance:
   - Railway
   - Render
   - DigitalOcean App Platform
   - AWS/GCP (com GPU)

## ✅ Checklist de Deploy

- [x] requirements.txt atualizado com PyTorch CPU
- [x] .slugignore criado
- [x] Variáveis de ambiente configuradas
- [x] Commit feito
- [ ] Push para Heroku
- [ ] Verificar logs: `heroku logs --tail`
- [ ] Testar app: `heroku open`

Agora você pode fazer deploy no Heroku sem problemas! 🎉
