// Add this at the very top of the file
console.log('experiment.js loaded');

// Add this at the beginning of the file
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded'); // Debug log
    
    // Ensure experiment tab is active
    const experimentTab = document.getElementById('experiment-tab');
    const basicTab = document.getElementById('basic-tab');
    const experimentPane = document.getElementById('experiment');
    const basicPane = document.getElementById('basic');
    
    if (experimentTab && basicTab && experimentPane && basicPane) {
        experimentTab.classList.add('active');
        basicTab.classList.remove('active');
        experimentPane.classList.add('show', 'active');
        basicPane.classList.remove('show', 'active');
    }

    // Handle outcome addition
    const addOutcomeBtn = document.getElementById('addOutcome');
    if (addOutcomeBtn) {
        addOutcomeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Add outcome clicked');
            window.addOutcome();
        });
    }

    // Handle outcome removal
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('remove-outcome')) {
            e.preventDefault();
            window.removeOutcome(e.target);
        }
    });

    // Handle experiment form submission
    const experimentForm = document.getElementById('experimentForm');
    console.log('Form element:', experimentForm); // Debug log
    
    if (experimentForm) {
        experimentForm.addEventListener('submit', function(e) {
            console.log('Form submitted');
            e.preventDefault();
            e.stopPropagation();
            
            // Disable the submit button
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            
            // Reset progress indicators
            document.getElementById('currentStatus').textContent = 'Running';
            document.getElementById('currentExperiment').textContent = document.getElementById('experimentName').value;
            document.getElementById('overallProgress').style.width = '0%';
            document.getElementById('overallProgress').textContent = '0%';
            document.getElementById('simulationProgressBar').style.width = '0%';
            document.getElementById('simulationProgressBar').textContent = '0%';
            
            // Collect form data
            const formData = {
                name: document.getElementById('experimentName').value,
                query: document.getElementById('experimentQuery').value,
                steps: parseInt(document.getElementById('experimentSteps').value),
                num_simulations: parseInt(document.getElementById('numSimulations').value),
                agent_type: document.getElementById('experimentAgentType').value,
                chunk_size: parseInt(document.getElementById('chunkSize').value),
                plot_results: document.getElementById('plotResults').checked,
                outcomes: Array.from(document.querySelectorAll('.outcome-item')).map(item => ({
                    name: item.querySelector('input:nth-child(2)').value,
                    condition: item.querySelector('input:nth-child(4)').value,
                    description: item.querySelector('input:nth-child(6)').value
                }))
            };

            console.log('Form data:', formData);

            // Create EventSource for server-sent events
            const eventSource = new EventSource(`/run_experiment?data=${encodeURIComponent(JSON.stringify(formData))}`);
            console.log('EventSource created'); // Debug log

            eventSource.onmessage = function(event) {
                console.log('Received message:', event.data);
                const data = JSON.parse(event.data);
                
                if (data.error) {
                    document.getElementById('currentStatus').textContent = 'Error';
                    console.error('Error:', data.error);
                    eventSource.close();
                    submitButton.disabled = false;
                    return;
                }

                if (data.progress) {
                    console.log('Progress update:', data.progress);
                    // Update simulation progress
                    const simulationProgress = data.progress;
                    document.getElementById('currentSimulation').textContent = 
                        `Simulation ${simulationProgress.current_step} of ${simulationProgress.total_steps}`;
                    document.getElementById('simulationProgressBar').style.width = 
                        `${simulationProgress.percentage}%`;
                    document.getElementById('simulationProgressBar').textContent = 
                        `${Math.round(simulationProgress.percentage)}%`;
                    
                    // Update overall progress
                    const overallProgress = (simulationProgress.current_step / simulationProgress.total_steps) * 100;
                    document.getElementById('overallProgress').style.width = `${overallProgress}%`;
                    document.getElementById('overallProgress').textContent = 
                        `${Math.round(overallProgress)}%`;
                }

                if (data.final_result) {
                    console.log('Final result received');
                    document.getElementById('currentStatus').textContent = 'Completed';
                    console.log('Final Result:', data.final_result);
                    eventSource.close();
                    submitButton.disabled = false;
                }
            };

            eventSource.onerror = function(error) {
                console.error('EventSource error:', error);
                document.getElementById('currentStatus').textContent = 'Error';
                eventSource.close();
                submitButton.disabled = false;
            };
        });
    } else {
        console.error('Experiment form not found!'); // Debug log
    }
});

// Make functions globally available
window.addOutcome = function() {
    console.log('addOutcome called');
    const outcomesContainer = document.getElementById('outcomesContainer');
    if (outcomesContainer) {
        const outcomeIndex = outcomesContainer.children.length;
        const outcomeHtml = `
            <div class="outcome-item mb-3">
                <div class="input-group">
                    <span class="input-group-text">Name</span>
                    <input type="text" class="form-control" placeholder="Outcome name">
                    <span class="input-group-text">Condition</span>
                    <input type="text" class="form-control" placeholder="Condition">
                    <span class="input-group-text">Description</span>
                    <input type="text" class="form-control" placeholder="Description">
                    <button class="btn btn-outline-danger remove-outcome" type="button">×</button>
                </div>
            </div>
        `;
        outcomesContainer.insertAdjacentHTML('beforeend', outcomeHtml);
        console.log('Outcome added');
    }
};

window.removeOutcome = function(button) {
    console.log('removeOutcome called');
    button.closest('.outcome-item').remove();
};

// Add this new function to show current outcomes count
window.updateOutcomesCount = function() {
    const count = document.querySelectorAll('.outcome-item').length;
    const countDisplay = document.getElementById('outcomesCount') || createOutcomesCount();
    countDisplay.textContent = `Current Outcomes: ${count}`;
};

// Helper function to create the outcomes count display
function createOutcomesCount() {
    const countDisplay = document.createElement('div');
    countDisplay.id = 'outcomesCount';
    countDisplay.className = 'mb-2 text-muted';
    document.getElementById('outcomesContainer').parentNode.insertBefore(countDisplay, document.getElementById('outcomesContainer'));
    return countDisplay;
}

// Initialize outcomes count when the page loads
document.addEventListener('DOMContentLoaded', function() {
    updateOutcomesCount();
});

// Load example configuration
window.loadExampleConfig = function() {
    const exampleConfig = {
        name: "prisoners_dilemma_experiment",
        query: "Two prisoners, Alice and Bob, are arrested for a crime. They are held in separate cells and cannot communicate. Each prisoner must decide whether to cooperate with the other by remaining silent or defect by betraying the other. If both cooperate, they each serve 1 year. If both defect, they each serve 2 years. If one defects and the other cooperates, the defector goes free and the cooperator serves 3 years. Simulate this scenario.",
        steps: 5,
        num_simulations: 3,
        results_folder: "results_prisoners_dilemma",
        agent_type: "timescale_aware",
        chunk_size: 1200,
        plot_results: true,
        outcomes: [
            {
                name: "both_cooperate",
                condition: "Both prisoners choose to cooperate and remain silent, resulting in equal outcomes of 1 year each",
                description: "Both prisoners serve 1 year in prison"
            },
            {
                name: "both_defect",
                condition: "Both prisoners choose to defect and betray each other, resulting in equal outcomes of 2 years each",
                description: "Both prisoners serve 2 years in prison"
            },
            {
                name: "alice_defects_bob_cooperates",
                condition: "Alice defects while Bob cooperates, resulting in Alice going free and Bob serving 3 years",
                description: "Alice goes free while Bob serves 3 years"
            },
            {
                name: "bob_defects_alice_cooperates",
                condition: "Bob defects while Alice cooperates, resulting in Bob going free and Alice serving 3 years",
                description: "Bob goes free while Alice serves 3 years"
            }
        ]
    };

    // Fill in the form fields
    document.getElementById('experimentName').value = exampleConfig.name;
    document.getElementById('experimentQuery').value = exampleConfig.query;
    document.getElementById('experimentSteps').value = exampleConfig.steps;
    document.getElementById('numSimulations').value = exampleConfig.num_simulations;
    document.getElementById('resultsFolder').value = exampleConfig.results_folder;
    document.getElementById('experimentAgentType').value = exampleConfig.agent_type;
    document.getElementById('chunkSize').value = exampleConfig.chunk_size;
    document.getElementById('plotResults').checked = exampleConfig.plot_results;

    // Clear existing outcomes and add new ones
    const outcomesContainer = document.getElementById('outcomesContainer');
    outcomesContainer.innerHTML = '';
    
    exampleConfig.outcomes.forEach(outcome => {
        const newOutcome = document.createElement('div');
        newOutcome.className = 'outcome-item mb-2';
        newOutcome.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <label class="form-label">Name</label>
                    <input type="text" class="form-control" value="${outcome.name}" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Condition</label>
                    <input type="text" class="form-control" value="${outcome.condition}" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Description</label>
                    <input type="text" class="form-control" value="${outcome.description}" required>
                </div>
                <div class="col-md-1 d-flex align-items-end">
                    <button type="button" class="btn btn-sm btn-danger" onclick="removeOutcome(this)">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `;
        outcomesContainer.appendChild(newOutcome);
    });
};

window.previewOutcomes = function() {
    const outcomes = Array.from(document.querySelectorAll('.outcome-item')).map(item => ({
        name: item.querySelector('input:nth-child(2)').value,
        condition: item.querySelector('input:nth-child(4)').value,
        description: item.querySelector('input:nth-child(6)').value
    }));
    
    console.log('Current Outcomes:', outcomes);
    alert(`Current Outcomes:\n${JSON.stringify(outcomes, null, 2)}`);
};

// Test if DOMContentLoaded is firing
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded event fired');
    
    // Test if we can find the form
    const form = document.getElementById('experimentForm');
    console.log('Form found:', form);
    
    // Test if we can find the add outcome button
    const addOutcomeBtn = document.getElementById('addOutcome');
    console.log('Add outcome button found:', addOutcomeBtn);
    
    // Test if we can find the outcomes container
    const outcomesContainer = document.getElementById('outcomesContainer');
    console.log('Outcomes container found:', outcomesContainer);
    
    // Add a test button click handler
    if (addOutcomeBtn) {
        addOutcomeBtn.addEventListener('click', function() {
            console.log('Add outcome button clicked');
            alert('Add outcome button clicked!'); // This will show a popup
        });
    }
    
    // Add a test form submit handler
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('Form submitted');
            alert('Form submitted!'); // This will show a popup
        });
    }
}); 