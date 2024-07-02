import React, { useState } from 'react';
import { Tabs, Tab, Box, Typography, Card, CardContent } from '@mui/material';

const models = [
  {
    id: 'neural-networks',
    name: 'Neural Networks',
    content: [
      "Inspired by biological neural networks, artificial neural networks are composed of interconnected nodes (neurons) organized in layers.",
      "Key components:",
      "• Input layer: Receives initial data",
      "• Hidden layer(s): Processes information",
      "• Output layer: Produces final result",
      "Example of a simple feedforward neural network:",
      "Given input x, hidden layer h, and output y:",
      "h = σ(W1x + b1)",
      "y = σ(W2h + b2)",
      "where W1, W2 are weight matrices, b1, b2 are bias vectors, and σ is an activation function",
      "Common activation functions:",
      "• ReLU: f(x) = max(0, x)",
      "• Sigmoid: σ(x) = 1 / (1 + e^(-x))",
      "• Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
      "Learning process:",
      "1. Forward propagation: Input data flows through the network",
      "2. Loss calculation: Measure error between prediction and true value",
      "3. Backpropagation: Compute gradients of the loss with respect to weights",
      "4. Weight update: Adjust weights to minimize loss",
      "Weight update formula: w_new = w_old - η * ∂L/∂w",
      "where η is the learning rate and ∂L/∂w is the gradient of the loss with respect to the weight",
      "Deep learning involves neural networks with many hidden layers, enabling the learning of complex hierarchical features"
    ]
  },
  {
    id: 'svm',
    name: 'Support Vector Machines (SVM)',
    content: [
      "SVMs are powerful supervised learning models used for classification and regression tasks.",
      "Key concept: Find the hyperplane that best separates classes in high-dimensional space.",
      "For linearly separable data:",
      "• Objective: Maximize the margin between classes",
      "• Margin is the distance between the hyperplane and the nearest data point from either class",
      "SVM optimization problem:",
      "Minimize: (1/2)||w||^2",
      "Subject to: y_i(w^T x_i + b) ≥ 1 for all i",
      "where w is the normal vector to the hyperplane, x_i are the training examples, and y_i are their labels",
      "Kernel Trick:",
      "• Used for non-linearly separable data",
      "• Maps input data to a higher-dimensional feature space where it becomes linearly separable",
      "• Allows SVM to find non-linear decision boundaries in the original input space",
      "• Kernel function K(x, x') computes the dot product in the higher-dimensional space without explicitly computing the mapping",
      "Common kernels:",
      "• Linear: K(x, x') = x^T x'",
      "• Polynomial: K(x, x') = (γx^T x' + r)^d",
      "• Radial Basis Function (RBF): K(x, x') = exp(-γ||x - x'||^2)",
      "Soft margin SVM introduces slack variables to allow for misclassifications:",
      "Minimize: (1/2)||w||^2 + C Σξ_i",
      "Subject to: y_i(w^T x_i + b) ≥ 1 - ξ_i and ξ_i ≥ 0 for all i",
      "where C is the regularization parameter and ξ_i are slack variables",
      "SVMs can be extended to multi-class classification using one-vs-rest or one-vs-one strategies"
    ]
  },
  {
    id: 'kmeans',
    name: 'K-means Clustering',
    content: [
      "K-means is an unsupervised learning algorithm used for partitioning n observations into k clusters.",
      "Objective: Minimize the within-cluster sum of squares (WCSS)",
      "WCSS = Σ Σ ||x - μ_i||^2",
      "where x is a data point and μ_i is the centroid of its cluster",
      "Algorithm steps:",
      "1. Initialize k centroids randomly",
      "2. Assign each data point to the nearest centroid",
      "3. Recalculate centroids as the mean of all points in the cluster",
      "4. Repeat steps 2-3 until convergence or maximum iterations reached",
      "Convergence is reached when centroids no longer move significantly",
      "Choosing optimal k:",
      "• Elbow method: Plot WCSS vs. k and look for the 'elbow' point",
      "• Silhouette analysis: Measure how similar an object is to its own cluster compared to other clusters",
      "Limitations:",
      "• Sensitive to initial centroid placement",
      "• Assumes spherical clusters of similar size",
      "• May converge to local optima",
      "Variations:",
      "• K-means++: Improves initial centroid selection",
      "• Mini-batch K-means: Uses subsets of data for faster processing on large datasets",
      "Distance metric is typically Euclidean, but other metrics can be used for specific applications"
    ]
  },
  {
    id: 'reinforcement-learning',
    name: 'Reinforcement Learning',
    content: [
      "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment.",
      "Key components:",
      "• Agent: The decision-maker",
      "• Environment: The world in which the agent operates",
      "• State (S): Current situation of the agent",
      "• Action (A): A decision made by the agent",
      "• Reward (R): Feedback from the environment",
      "• Policy (π): Strategy that the agent employs to determine actions",
      "Objective: Maximize cumulative reward over time",
      "Value function: V(s) = E[Σ γ^t R_t | S_0 = s]",
      "where γ is the discount factor and R_t is the reward at time t",
      "Q-function (Action-Value function):",
      "Q(s,a) = E[Σ γ^t R_t | S_0 = s, A_0 = a]",
      "• Represents the expected return starting from state s, taking action a, and then following policy π",
      "• Helps in determining the best action to take in a given state",
      "Bellman equation:",
      "V(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]",
      "• Expresses the value of a state in terms of the values of its successor states",
      "• R(s,a) is the immediate reward, P(s'|s,a) is the transition probability to state s' from state s taking action a",
      "• Forms the basis for many RL algorithms, including value iteration and Q-learning",
      "Common RL algorithms:",
      "• Q-learning: Learn the optimal Q-function",
      "• Policy Gradient: Directly optimize the policy",
      "• Actor-Critic: Combine value function approximation with policy optimization",
      "Exploration vs Exploitation:",
      "• Exploration: Trying new actions to potentially find better strategies",
      "• Exploitation: Using known information to maximize reward",
      "• Common approach: ε-greedy (choose random action with probability ε)",
      "Challenges in RL:",
      "• Credit assignment problem: Determining which actions led to rewards",
      "• Sample efficiency: Learning from limited interactions",
      "• Stability and convergence issues in function approximation methods"
    ]
  },
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
    ],
    image: "./RNN.webp"
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
    ],
    image: "./LSTM.jpg"
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
    ],
    image: "./Transformer.webp"
  }
];

const ModelContent = ({ content, image }) => (
  <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
    <Card sx={{ flex: 1 }}>
      <CardContent>
        <ul style={{ paddingLeft: '20px' }}>
          {content.map((point, index) => (
            <li key={index} style={{ fontSize: '0.9rem', marginBottom: '8px' }}>
              {point.startsWith('•') ? (
                <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
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
    <Box sx={{ flex: 1 }}>
      <img src={image} alt="Model architecture" style={{ width: '100%', height: 'auto' }} />
    </Box>
  </Box>
);

const SequenceModelsDashboard = () => {
  const [activeTab, setActiveTab] = useState(0);

  const handleChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Sequence-based Deep Learning Models: Technical Overview
      </Typography>
      <Tabs value={activeTab} onChange={handleChange} aria-label="model tabs">
        {models.map((model, index) => (
          <Tab key={model.id} label={model.name} id={`tab-${index}`} aria-controls={`tabpanel-${index}`} />
        ))}
      </Tabs>
      {models.map((model, index) => (
        <div
          key={model.id}
          role="tabpanel"
          hidden={activeTab !== index}
          id={`tabpanel-${index}`}
          aria-labelledby={`tab-${index}`}
        >
          {activeTab === index && (
            <Box sx={{ py: 3 }}>
              <Typography variant="h5" component="h2" gutterBottom>
                {model.name}
              </Typography>
              <ModelContent content={model.content} image={model.image} />
            </Box>
          )}
        </div>
      ))}
    </Box>
  );
};

export default SequenceModelsDashboard;