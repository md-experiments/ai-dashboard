import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent } from '@/components/ui/card';

const models = [
  {
    id: 'rnn',
    name: 'Recurrent Neural Networks (RNN)',
    content: [
      "RNNs are designed to work with sequence data by maintaining a hidden state that captures information from previous time steps.",
      "Basic RNN architecture:",
      "h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)",
      "y_t = W_hy * h_t + b_y",
      "where:",
      "• h_t is the hidden state at time t",
      "• x_t is the input at time t",
      "• y_t is the output at time t",
      "• W_hh, W_xh, W_hy are weight matrices",
      "• b_h, b_y are bias vectors",
      "Key characteristics:",
      "• Shares parameters across all time steps",
      "• Can process sequences of variable length",
      "• Capable of capturing temporal dependencies",
      "Training RNNs:",
      "• Use Backpropagation Through Time (BPTT)",
      "• Compute gradients by unrolling the network over time steps",
      "Challenges:",
      "• Vanishing/exploding gradients over long sequences",
      "• Difficulty in capturing long-term dependencies",
      "Applications:",
      "• Language modeling",
      "• Speech recognition",
      "• Time series prediction"
    ]
  },
  {
    id: 'lstm',
    name: 'Long Short-Term Memory (LSTM)',
    content: [
      "LSTMs are a type of RNN designed to overcome the vanishing gradient problem and better capture long-term dependencies.",
      "LSTM cell architecture consists of three gates: forget gate, input gate, and output gate.",
      "Key equations:",
      "f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  // Forget gate",
      "i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  // Input gate",
      "C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  // Candidate cell state",
      "C_t = f_t * C_{t-1} + i_t * C̃_t  // Cell state update",
      "o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  // Output gate",
      "h_t = o_t * tanh(C_t)  // Hidden state",
      "where:",
      "• σ is the sigmoid function",
      "• * denotes element-wise multiplication",
      "• [h_{t-1}, x_t] represents concatenation of h_{t-1} and x_t",
      "Key features:",
      "• Ability to selectively remember or forget information",
      "• Better gradient flow through the network",
      "• More effective at capturing long-term dependencies",
      "Variants:",
      "• Peephole connections: Allow gates to look at the cell state",
      "• Gated Recurrent Unit (GRU): Simplified version with fewer parameters",
      "Applications:",
      "• Machine translation",
      "• Sentiment analysis",
      "• Music generation"
    ]
  },
  {
    id: 'transformers',
    name: 'Transformers',
    content: [
      "Transformers are a type of neural network architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions.",
      "Key components:",
      "1. Self-Attention Mechanism:",
      "   Attention(Q, K, V) = softmax(QK^T / √d_k)V",
      "   where Q (query), K (key), and V (value) are linear projections of the input",
      "2. Multi-Head Attention:",
      "   Allows the model to jointly attend to information from different representation subspaces",
      "3. Position Encoding:",
      "   Injects information about the relative or absolute position of tokens in the sequence",
      "4. Feed-Forward Networks:",
      "   Applied to each position separately and identically",
      "Encoder-Decoder Architecture:",
      "• Encoder: Processes the input sequence",
      "• Decoder: Generates the output sequence",
      "Training objective often uses masked language modeling and next sentence prediction",
      "Key advantages:",
      "• Parallelizable: Can process entire sequences simultaneously",
      "• Captures long-range dependencies effectively",
      "• Scales well to very large datasets and model sizes",
      "Variants:",
      "• BERT: Bidirectional Encoder Representations from Transformers",
      "• GPT: Generative Pre-trained Transformer",
      "• T5: Text-to-Text Transfer Transformer",
      "Applications:",
      "• Natural language processing tasks (translation, summarization, question answering)",
      "• Image recognition (Vision Transformer)",
      "• Speech recognition and synthesis"
    ]
  }
];

const ModelContent = ({ content }) => (
  <Card className="mt-4">
    <CardContent className="p-4">
      <ul className="space-y-2">
        {content.map((point, index) => (
          <li key={index} className="text-sm">
            {point.startsWith('•') ? (
              <ul className="list-disc pl-5">
                <li>{point.slice(2)}</li>
              </ul>
            ) : (
              point
            )}
          </li>
        ))}
      </ul>
    </CardContent>
  </Card>
);

const SequenceModelsDashboard = () => {
  const [activeTab, setActiveTab] = useState("rnn");

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Sequence-based Deep Learning Models: Technical Overview</h1>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          {models.map((model) => (
            <TabsTrigger key={model.id} value={model.id}>{model.name}</TabsTrigger>
          ))}
        </TabsList>
        {models.map((model) => (
          <TabsContent key={model.id} value={model.id}>
            <h2 className="text-xl font-semibold mb-2">{model.name}</h2>
            <ModelContent content={model.content} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
};

export default SequenceModelsDashboard;