import React, { useState } from 'react';
import { Box, Button } from '@mui/material';
import FinancialRiskManagementTable from './UseCases';
import SequenceModelsDashboard from './Models';

const CombinedDashboard = () => {
  const [showMLModels, setShowMLModels] = useState(true);

  const toggleDashboard = () => setShowMLModels(!showMLModels);

  return (
    <Box sx={{ position: 'relative', padding: 2 }}>
      <Button 
        onClick={toggleDashboard}
        sx={{ position: 'absolute', top: 16, right: 16, zIndex: 1 }}
        variant="contained"
      >
        Switch to {showMLModels ? 'Financial Use Cases' : 'AI/ML Models'}
      </Button>
      {showMLModels ? <SequenceModelsDashboard /> : <FinancialRiskManagementTable />}
    </Box>
  );
};

export default CombinedDashboard;