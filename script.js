// Meme Coin Profit Calculator JavaScript
class MemeCoinCalculator {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.hideResults();
    }

    initializeElements() {
        // Input elements
        this.initialMcpInput = document.getElementById('initialMcp');
        this.finalMcpInput = document.getElementById('finalMcp');
        this.solInvestmentInput = document.getElementById('solInvestment');
        
        // Button elements
        this.calculateBtn = document.getElementById('calculateBtn');
        this.resetBtn = document.getElementById('resetBtn');
        
        // Result elements
        this.resultsSection = document.getElementById('resultsSection');
        this.profitLossElement = document.getElementById('profitLoss');
        this.percentageGainElement = document.getElementById('percentageGain');
        this.finalValueElement = document.getElementById('finalValue');
        this.roiElement = document.getElementById('roi');
        this.mcpMultiplierElement = document.getElementById('mcpMultiplier');
        
        // Result cards for styling
        this.resultCards = document.querySelectorAll('.result-card');
    }

    attachEventListeners() {
        // Real-time calculation on input
        this.initialMcpInput.addEventListener('input', () => this.handleInput(this.initialMcpInput));
        this.finalMcpInput.addEventListener('input', () => this.handleInput(this.finalMcpInput));
        this.solInvestmentInput.addEventListener('input', () => this.handleInput(this.solInvestmentInput));

        // Button listeners
        this.calculateBtn.addEventListener('click', () => this.calculateProfits());
        this.resetBtn.addEventListener('click', () => this.resetCalculator());

        // Real-time calculation when all fields have values
        [this.initialMcpInput, this.finalMcpInput, this.solInvestmentInput].forEach(input => {
            input.addEventListener('input', () => this.debounceCalculation());
        });
    }

    handleInput(input) {
        // Format number as user types
        const rawValue = input.value.replace(/[^0-9.]/g, '');
        const numericValue = parseFloat(rawValue);
        
        if (!isNaN(numericValue) && rawValue !== '') {
            input.value = this.formatNumber(numericValue);
            this.validateInput(input);
        } else if (rawValue === '') {
            input.value = '';
            this.clearValidation(input);
        }
    }

    validateInput(input) {
        const value = this.parseNumber(input.value);
        
        if (value > 0) {
            input.classList.remove('error');
            input.classList.add('valid');
            return true;
        } else {
            input.classList.remove('valid');
            input.classList.add('error');
            return false;
        }
    }

    clearValidation(input) {
        input.classList.remove('valid', 'error');
    }

    debounceCalculation() {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            if (this.areAllInputsValid()) {
                this.calculateProfits(true); // Auto-calculate
            }
        }, 500);
    }

    areAllInputsValid() {
        const initialMcp = this.parseNumber(this.initialMcpInput.value);
        const finalMcp = this.parseNumber(this.finalMcpInput.value);
        const solInvestment = this.parseNumber(this.solInvestmentInput.value);
        
        return initialMcp > 0 && finalMcp > 0 && solInvestment > 0;
    }

    calculateProfits(isAutoCalculation = false) {
        if (!this.areAllInputsValid()) {
            this.showValidationErrors();
            return;
        }

        if (!isAutoCalculation) {
            this.showCalculatingState();
        }

        setTimeout(() => {
            const results = this.performCalculations();
            this.displayResults(results);
            this.showResults();
            
            if (!isAutoCalculation) {
                this.hideCalculatingState();
            }
        }, isAutoCalculation ? 0 : 800);
    }

    performCalculations() {
        const initialMcp = this.parseNumber(this.initialMcpInput.value);
        const finalMcp = this.parseNumber(this.finalMcpInput.value);
        const solInvestment = this.parseNumber(this.solInvestmentInput.value);

        // Calculate market cap multiplier
        const mcpMultiplier = finalMcp / initialMcp;
        
        // Calculate final portfolio value
        const finalPortfolioValue = solInvestment * mcpMultiplier;
        
        // Calculate profit/loss
        const profitLoss = finalPortfolioValue - solInvestment;
        
        // Calculate percentage gain/loss
        const percentageGain = ((finalPortfolioValue - solInvestment) / solInvestment) * 100;
        
        // ROI is the same as percentage gain
        const roi = percentageGain;

        return {
            initialMcp,
            finalMcp,
            solInvestment,
            mcpMultiplier,
            finalPortfolioValue,
            profitLoss,
            percentageGain,
            roi,
            isProfit: profitLoss >= 0
        };
    }

    displayResults(results) {
        // Update result values with animation
        this.animateValueUpdate(this.profitLossElement, 
            `◎${this.formatNumber(Math.abs(results.profitLoss))}`);
        
        this.animateValueUpdate(this.percentageGainElement, 
            `${results.percentageGain >= 0 ? '+' : ''}${results.percentageGain.toFixed(2)}%`);
        
        this.animateValueUpdate(this.finalValueElement, 
            `◎${this.formatNumber(results.finalPortfolioValue)}`);
        
        this.animateValueUpdate(this.roiElement, 
            `${results.roi >= 0 ? '+' : ''}${results.roi.toFixed(2)}%`);
        
        this.animateValueUpdate(this.mcpMultiplierElement, 
            `${results.mcpMultiplier.toFixed(2)}x`);

        // Update styling based on profit/loss
        this.updateResultStyling(results.isProfit);
    }

    animateValueUpdate(element, newValue) {
        element.classList.add('updating');
        setTimeout(() => {
            element.textContent = newValue;
            element.classList.remove('updating');
        }, 150);
    }

    updateResultStyling(isProfit) {
        // Update profit/loss indicator
        const profitIcon = document.querySelector('.result-icon.profit, .result-icon.loss');
        const profitIconContainer = profitIcon.parentElement.querySelector('.result-icon');
        
        profitIconContainer.className = `result-icon ${isProfit ? 'profit' : 'loss'}`;
        profitIcon.innerHTML = `<i class="fas fa-trending-${isProfit ? 'up' : 'down'}"></i>`;

        // Update text colors
        [this.profitLossElement, this.percentageGainElement, this.roiElement].forEach(element => {
            element.classList.remove('profit', 'loss');
            element.classList.add(isProfit ? 'profit' : 'loss');
        });
    }

    showCalculatingState() {
        this.calculateBtn.classList.add('calculating');
        this.calculateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating...';
    }

    hideCalculatingState() {
        this.calculateBtn.classList.remove('calculating');
        this.calculateBtn.innerHTML = '<i class="fas fa-calculator"></i> Calculate Profit';
    }

    showValidationErrors() {
        [this.initialMcpInput, this.finalMcpInput, this.solInvestmentInput].forEach(input => {
            const value = this.parseNumber(input.value);
            if (value <= 0 || isNaN(value)) {
                input.classList.add('error');
                input.classList.remove('valid');
            }
        });

        // Shake animation for calculate button
        this.calculateBtn.style.animation = 'shake 0.5s ease-in-out';
        setTimeout(() => {
            this.calculateBtn.style.animation = '';
        }, 500);
    }

    showResults() {
        this.resultsSection.classList.add('show');
        
        // Animate result cards
        this.resultCards.forEach((card, index) => {
            card.style.animation = `slide-up 0.6s ease-out ${index * 0.1}s both`;
        });
    }

    hideResults() {
        this.resultsSection.classList.remove('show');
    }

    resetCalculator() {
        // Clear inputs
        this.initialMcpInput.value = '';
        this.finalMcpInput.value = '';
        this.solInvestmentInput.value = '';

        // Clear validations
        [this.initialMcpInput, this.finalMcpInput, this.solInvestmentInput].forEach(input => {
            this.clearValidation(input);
        });

        // Hide results
        this.hideResults();

        // Reset button animation
        this.resetBtn.style.transform = 'rotate(360deg)';
        setTimeout(() => {
            this.resetBtn.style.transform = '';
        }, 300);

        // Focus on first input
        this.initialMcpInput.focus();
    }

    formatNumber(num) {
        if (num >= 1000000000) {
            return (num / 1000000000).toFixed(2) + 'B';
        } else if (num >= 1000000) {
            return (num / 1000000).toFixed(2) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(2) + 'K';
        } else {
            return num.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }
    }

    parseNumber(str) {
        if (!str) return 0;
        
        // Remove formatting characters
        const cleaned = str.replace(/[,$BMK]/g, '');
        let num = parseFloat(cleaned);
        
        // Handle suffixes
        if (str.includes('B')) {
            num *= 1000000000;
        } else if (str.includes('M')) {
            num *= 1000000;
        } else if (str.includes('K')) {
            num *= 1000;
        }
        
        return isNaN(num) ? 0 : num;
    }
}

// Initialize calculator when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MemeCoinCalculator();
});

// Add shake animation to CSS dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-2px); }
        20%, 40%, 60%, 80% { transform: translateX(2px); }
    }
`;
document.head.appendChild(style);