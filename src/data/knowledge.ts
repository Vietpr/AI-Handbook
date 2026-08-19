export const KNOWLEDGE = [
  {
    id: 'foundations', index: '01', title: 'Foundations', viTitle: 'Nền tảng', short: 'Math · probability · optimization', viShort: 'Toán · xác suất · tối ưu',
    body: 'The mathematical language needed to reason about data, models and learning.',
    viBody: 'Những nền tảng toán học cần thiết để hiểu dữ liệu, mô hình và quá trình học.',
    topics: ['Vectors & matrices', 'Probability', 'Statistics', 'Optimization', 'Representation'],
  },
  {
    id: 'machine-learning', index: '02', title: 'Machine Learning', viTitle: 'Machine Learning', short: 'Models · features · evaluation', viShort: 'Mô hình · đặc trưng · đánh giá',
    body: 'Learn what it means for a system to generalize before moving to larger neural models.',
    viBody: 'Hiểu một mô hình học và khái quát hóa như thế nào trước khi đi tới các mô hình neural lớn hơn.',
    topics: ['Learning', 'Regression', 'Classification', 'Trees', 'Regularization', 'Evaluation'],
  },
  {
    id: 'computer-vision', index: '03', title: 'Computer Vision', viTitle: 'Computer Vision', short: 'Pixels · detection · perception', viShort: 'Pixel · detection · perception',
    body: 'How visual structure becomes representations, predictions and decisions.',
    viBody: 'Cách dữ liệu hình ảnh được biểu diễn, phân tích và chuyển thành dự đoán.',
    topics: ['Image processing', 'CNNs', 'Detection', 'Segmentation', 'OCR', 'Tracking', 'Vision transformers'],
  },
  {
    id: 'generative-ai', index: '04', title: 'Generative AI', viTitle: 'Generative AI', short: 'Transformers · RAG · agents', viShort: 'Transformer · RAG · agent',
    body: 'From tokens and attention to context engineering, retrieval, tools and agents.',
    viBody: 'Từ token, attention tới context engineering, retrieval, tool use và agent.',
    topics: ['Tokenization', 'Attention', 'Transformers', 'LLMs', 'RAG', 'Context engineering', 'Agents', 'Evaluation'],
  },
  {
    id: 'ai-systems', index: '05', title: 'AI Systems', viTitle: 'AI Systems', short: 'Search · serving · evaluation · scale', viShort: 'Search · serving · đánh giá · scale',
    body: 'The engineering layer that turns model behavior into a dependable product.',
    viBody: 'Lớp engineering biến khả năng của mô hình thành một hệ thống có thể vận hành ổn định.',
    topics: ['Data pipelines', 'Retrieval', 'Serving', 'Caching', 'Observability', 'Latency', 'Cost', 'Scaling'],
  },
] as const;
