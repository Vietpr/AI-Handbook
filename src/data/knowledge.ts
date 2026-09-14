/*
 * The six Learn chapters. `body` / `viBody` is the one-line description shown
 * on the Learn index and at the top of each chapter page - plain words about
 * what the articles inside cover. Edit freely.
 */
export const KNOWLEDGE = [
  {
    id: 'foundations', index: '01', title: 'Foundations', viTitle: 'Nền tảng',
    body: 'The math and mental models you need to understand how models learn, generalize, and fail.',
    viBody: 'Phần toán và các mô hình tư duy cần thiết để hiểu cách mô hình học, khái quát hóa và thất bại.',
  },
  {
    id: 'machine-learning', index: '02', title: 'Machine Learning', viTitle: 'Machine Learning',
    body: 'Classic algorithms worked through on real data with scikit-learn, from linear models to trees, clustering, and end-to-end projects.',
    viBody: 'Các thuật toán kinh điển được triển khai trên dữ liệu thật bằng scikit-learn, từ mô hình tuyến tính đến cây, phân cụm và các project end-to-end.',
  },
  {
    id: 'deep-learning', index: '03', title: 'Deep Learning', viTitle: 'Deep Learning',
    body: 'Neural networks from first principles to practical PyTorch training, including CNNs, RNNs, regularization, and transfer learning.',
    viBody: 'Mạng nơ-ron từ nguyên lý nền tảng đến huấn luyện thực tế với PyTorch, gồm CNN, RNN, regularization và transfer learning.',
  },
  {
    id: 'computer-vision', index: '04', title: 'Computer Vision', viTitle: 'Computer Vision',
    body: 'How models learn from images, from pixels and CNNs to detection, segmentation, vision transformers, and CLIP.',
    viBody: 'Cách mô hình học từ hình ảnh, từ pixel và CNN đến detection, segmentation, vision transformer và CLIP.',
  },
  {
    id: 'generative-ai', index: '05', title: 'Generative AI', viTitle: 'Generative AI',
    body: 'How generative models work, from tokenization and attention to prompting, retrieval, diffusion, and RAG.',
    viBody: 'Cách các mô hình sinh hoạt động, từ tokenization và attention đến prompting, retrieval, diffusion và RAG.',
  },
  {
    id: 'ai-systems', index: '06', title: 'AI Systems', viTitle: 'AI Systems',
    body: 'How AI models are served and measured in practice: memory, batching, inference, caching, observability, and deployment.',
    viBody: 'Cách mô hình AI được triển khai và đo lường trong thực tế: bộ nhớ, batching, inference, caching, observability và deployment.',
  },
] as const;
