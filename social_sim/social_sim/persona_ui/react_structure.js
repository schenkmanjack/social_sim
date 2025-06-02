// App.js
import React, { useState } from 'react';
import './App.css';

const App = () => {
  const [currentPage, setCurrentPage] = useState('main');
  const [query, setQuery] = useState('');
  const [generatedPersonas, setGeneratedPersonas] = useState([]);
  const [selectedPersonas, setSelectedPersonas] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGeneratePersonas = async () => {
    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/generate-personas', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim()
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setGeneratedPersonas(result.personas);
        setCurrentPage('personas');
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error generating personas:', error);
      alert('Failed to generate personas. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePersonaSelect = (persona) => {
    setSelectedPersonas(prev => {
      const isSelected = prev.some(p => p.id === persona.id);
      if (isSelected) {
        return prev.filter(p => p.id !== persona.id);
      } else {
        return [...prev, persona];
      }
    });
  };

  const handleEvaluate = async () => {
    if (selectedPersonas.length === 0) {
      alert('Please select at least one persona');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          personas: selectedPersonas
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(result.results);
        setCurrentPage('results');
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error evaluating content:', error);
      alert('Failed to evaluate content. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (currentPage === 'main') {
    return (
      <MainPage 
        query={query}
        setQuery={setQuery}
        onGeneratePersonas={handleGeneratePersonas}
        loading={loading}
      />
    );
  }

  if (currentPage === 'personas') {
    return (
      <PersonasPage 
        personas={generatedPersonas}
        selectedPersonas={selectedPersonas}
        onPersonaSelect={handlePersonaSelect}
        onBack={() => setCurrentPage('main')}
        onEvaluate={handleEvaluate}
        loading={loading}
      />
    );
  }

  if (currentPage === 'results') {
    return (
      <ResultsPage 
        results={results}
        query={query}
        onStartOver={() => {
          setCurrentPage('main');
          setQuery('');
          setGeneratedPersonas([]);
          setSelectedPersonas([]);
          setResults(null);
        }}
      />
    );
  }
};

// MainPage.js
const MainPage = ({ query, setQuery, onGeneratePersonas, loading }) => {
  return (
    <div className="container">
      <div className="main-page">
        <h1>AI Persona Content Evaluator</h1>
        <p>Enter your content below and we'll generate AI personas to evaluate it</p>
        
        <div className="query-section">
          <div className="query-input-container">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter the content you want to evaluate..."
              className="query-input"
              rows={6}
            />
            <button 
              className="generate-btn"
              onClick={onGeneratePersonas}
              disabled={loading || !query.trim()}
            >
              {loading ? 'Generating...' : '→'}
            </button>
          </div>
        </div>
        
        <div className="simulation-options">
          <h3>Simulation Options</h3>
          <p>AI personas will be automatically generated based on your content</p>
        </div>
      </div>
    </div>
  );
};

// PersonasPage.js
const PersonasPage = ({ personas, selectedPersonas, onPersonaSelect, onBack, onEvaluate, loading }) => {
  return (
    <div className="container">
      <div className="personas-page">
        <button className="back-btn" onClick={onBack}>← Back</button>
        
        <div className="personas-header">
          <h1>Generated Personas</h1>
          <p>Select AI agents to evaluate your content</p>
        </div>

        <div className="personas-grid">
          {personas.map(persona => (
            <PersonaCard 
              key={persona.id}
              persona={persona}
              isSelected={selectedPersonas.some(p => p.id === persona.id)}
              onSelect={() => onPersonaSelect(persona)}
            />
          ))}
        </div>

        {selectedPersonas.length > 0 && (
          <div className="evaluate-section">
            <button 
              className="evaluate-btn" 
              onClick={onEvaluate}
              disabled={loading}
            >
              {loading ? 'Evaluating...' : `Evaluate with ${selectedPersonas.length} Persona${selectedPersonas.length !== 1 ? 's' : ''}`}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// ResultsPage.js
const ResultsPage = ({ results, query, onStartOver }) => {
  return (
    <div className="container">
      <div className="results-page">
        <h1>Evaluation Results</h1>
        <div className="results-content">
          {/* Render results content here */}
        </div>
        <button className="start-over-btn" onClick={onStartOver}>Start Over</button>
      </div>
    </div>
  );
};

export default App;