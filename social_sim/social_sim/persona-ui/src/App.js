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
  
    console.log('Starting persona generation with query:', query);
    setLoading(true);
    try {
      console.log('Making request to API...');
      const response = await fetch('http://localhost:5000/api/generate-personas', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim()
        })
      });
  
      console.log('Response status:', response.status);
      const result = await response.json();
      console.log('Full API response:', result);
      
      if (result.success) {
        console.log('Success! Setting personas:', result.personas);
        setGeneratedPersonas(result.personas);
        setCurrentPage('personas');
      } else {
        console.error('API returned error:', result.error);
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Network/parsing error:', error);
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
      const response = await fetch('http://localhost:5000/api/evaluate', {
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
  console.log('MainPage render - query:', query, 'loading:', loading, 'button disabled:', loading || !query.trim());
  
  return (
    <div className="container">
      <div className="main-page">
        <h1>AI Persona Content Evaluator</h1>
        <p>Enter your content below and we'll generate AI personas to evaluate it</p>
        
        <div className="query-section">
          <div className="query-input-container">
            <textarea
              value={query}
              onChange={(e) => {
                console.log('Textarea changed:', e.target.value);
                setQuery(e.target.value);
              }}
              placeholder="Enter the content you want to evaluate..."
              className="query-input"
              rows={6}
            />
            <button 
              className="generate-btn"
              onClick={() => {
                console.log('Arrow button clicked!');
                onGeneratePersonas();
              }}
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

// PersonaCard component
const PersonaCard = ({ persona, isSelected, onSelect }) => {
  return (
    <div 
      className={`persona-card ${isSelected ? 'selected' : ''}`}
      onClick={onSelect}
    >
      <div className="persona-avatar">{persona.avatar}</div>
      <div className="persona-name">{persona.name}</div>
      <div className="persona-description">{persona.description}</div>
    </div>
  );
};

// ResultsPage.js
const ResultsPage = ({ results, query, onStartOver }) => {
  return (
    <div className="container">
      <div className="results-page">
        <div className="results-header">
          <h1>Evaluation Results</h1>
          <p>Query: "{query}"</p>
        </div>

        <div className="results-content">
          {results && results.evaluations && results.evaluations.map((evaluation, index) => (
            <div key={index} className="persona-result">
              <h3>Evaluation {index + 1}</h3>
              <div className="evaluation-item">
                <p>{evaluation}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="results-actions">
          <button className="start-over-btn" onClick={onStartOver}>Start Over</button>
        </div>
      </div>
    </div>
  );
};

export default App;