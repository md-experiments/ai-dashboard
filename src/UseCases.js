import React, { useState } from 'react';
import { Box, Typography, Tabs, Tab, Card, CardContent, Grid } from '@mui/material';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const methods = ['Regression', 'Classification', 'Clustering', 'Neural Networks', 'Reinforcement Learning'];
const riskTypes = ['Credit Risk', 'Market Risk', 'Operational Risk', 'Liquidity Risk', 'Compliance Risk'];

const gridData = {
  'Credit Risk': {
    'Regression': {
      useCase: 'Credit Scoring',
      details: 'Predicting credit scores or probability of default using historical data and financial indicators.'
    },
    'Classification': {
      useCase: 'Loan Approval',
      details: 'Classifying loan applications as approve/reject based on applicant characteristics and credit history.'
    },
    'Clustering': {
      useCase: 'Customer Segmentation',
      details: 'Grouping borrowers with similar risk profiles for targeted risk management strategies.'
    },
    'Neural Networks': {
      useCase: 'Complex Default Prediction',
      details: 'Using deep learning to predict defaults based on complex patterns in large datasets.'
    },
    'Reinforcement Learning': {
      useCase: 'Dynamic Credit Limits',
      details: 'Adjusting credit limits in real-time based on customer behavior and market conditions.'
    }
  },
  'Market Risk': {
    'Regression': {
      useCase: 'Value at Risk (VaR) Estimation',
      details: 'Predicting potential losses in portfolio value over a specific time horizon.',
      formula: 'VaR_\\alpha = -\\inf\\{l: P(L > l) \\leq \\alpha\\}'
    },
    'Classification': {
      useCase: 'Market Regime Detection',
      details: 'Identifying different market regimes (e.g., bull, bear, volatile) to adjust risk strategies.'
    },
    'Clustering': {
      useCase: 'Asset Correlation Analysis',
      details: 'Grouping assets with similar risk characteristics for portfolio diversification.'
    },
    'Neural Networks': {
      useCase: 'Option Pricing',
      details: 'Using neural networks to price complex derivatives and assess related risks.',
      formula: 'C = S_0N(d_1) - Ke^{-rT}N(d_2)'
    },
    'Reinforcement Learning': {
      useCase: 'Dynamic Hedging Strategies',
      details: 'Developing adaptive hedging strategies that respond to changing market conditions.'
    }
  },
  'Operational Risk': {
    'Regression': {
      useCase: 'Loss Prediction',
      details: 'Estimating potential losses from operational failures based on historical data.'
    },
    'Classification': {
      useCase: 'Fraud Detection',
      details: 'Identifying potentially fraudulent transactions or activities in real-time.'
    },
    'Clustering': {
      useCase: 'Risk Event Categorization',
      details: 'Grouping similar operational risk events for targeted mitigation strategies.'
    },
    'Neural Networks': {
      useCase: 'Anomaly Detection',
      details: 'Detecting unusual patterns in operations data that may indicate risks or failures.'
    },
    'Reinforcement Learning': {
      useCase: 'Adaptive Control Systems',
      details: 'Developing systems that learn to respond to and mitigate operational risks over time.'
    }
  },
  'Liquidity Risk': {
    'Regression': {
      useCase: 'Cash Flow Forecasting',
      details: 'Predicting future cash flows to manage liquidity and avoid shortfalls.',
      formula: 'CF_t = \\beta_0 + \\beta_1X_{1t} + \\beta_2X_{2t} + ... + \\epsilon_t'
    },
    'Classification': {
      useCase: 'Liquidity Stress Testing',
      details: 'Classifying scenarios as high/medium/low liquidity risk for stress testing.'
    },
    'Clustering': {
      useCase: 'Funding Source Analysis',
      details: 'Grouping funding sources with similar stability characteristics.'
    },
    'Neural Networks': {
      useCase: 'Complex Liquidity Modeling',
      details: 'Modeling complex relationships between market conditions and liquidity needs.'
    },
    'Reinforcement Learning': {
      useCase: 'Dynamic Liquidity Management',
      details: 'Optimizing liquidity reserves and funding mix in response to changing conditions.'
    }
  },
  'Compliance Risk': {
    'Regression': {
      useCase: 'Regulatory Fine Prediction',
      details: 'Estimating potential fines based on compliance-related factors and historical data.'
    },
    'Classification': {
      useCase: 'AML Transaction Monitoring',
      details: 'Classifying transactions as potentially suspicious for anti-money laundering (AML) compliance.'
    },
    'Clustering': {
      useCase: 'Regulatory Requirement Grouping',
      details: 'Clustering similar regulatory requirements for efficient compliance management.'
    },
    'Neural Networks': {
      useCase: 'Complex Pattern Recognition in Compliance Data',
      details: 'Identifying complex patterns of non-compliance across large datasets.'
    },
    'Reinforcement Learning': {
      useCase: 'Adaptive Compliance Monitoring',
      details: 'Developing systems that learn and adapt to new compliance requirements and risks over time.'
    }
  }
};

const GridCell = ({ content }) => (
  <Card variant="outlined" sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" gutterBottom>
        {content.useCase}
      </Typography>
      <Typography variant="body2">
        {content.details}
      </Typography>
      {content.formula && (
        <Box mt={2}>
          <BlockMath math={content.formula} />
        </Box>
      )}
    </CardContent>
  </Card>
);

const FinancialRiskManagementTable = () => {
  const [activeTab, setActiveTab] = useState(0);

  const handleChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ width: '100%', typography: 'body1' }}>
      <Typography variant="h4" gutterBottom>
        AI in Financial Risk Management: Methods and Use Cases
      </Typography>
      <Tabs value={activeTab} onChange={handleChange} aria-label="risk types tabs">
        {riskTypes.map((riskType, index) => (
          <Tab key={riskType} label={riskType} id={`tab-${index}`} aria-controls={`tabpanel-${index}`} />
        ))}
      </Tabs>
      {riskTypes.map((riskType, index) => (
        <div
          key={riskType}
          role="tabpanel"
          hidden={activeTab !== index}
          id={`tabpanel-${index}`}
          aria-labelledby={`tab-${index}`}
        >
          {activeTab === index && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>{riskType}</Typography>
              <Grid container spacing={2}>
                {methods.map((method) => (
                  <Grid item xs={12} sm={6} md={4} key={method}>
                    <Typography variant="subtitle1" gutterBottom>{method}</Typography>
                    <GridCell content={gridData[riskType][method]} />
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
        </div>
      ))}
    </Box>
  );
};

export default FinancialRiskManagementTable;