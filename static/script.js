document.addEventListener("DOMContentLoaded", function () {
    const outputDiv = document.getElementById("typing-output");
    const resultsDiv = document.getElementById("results-data");
    const results = resultsDiv ? JSON.parse(resultsDiv.textContent || "[]") : []; // Parse JSON if available

    let lineIndex = 0;

    // Function to type out each line character by character
    function typeLine(line) {
        const lineDiv = document.createElement("div"); // Create a new line element
        outputDiv.appendChild(lineDiv);

        let charIndex = 0;

        function typeCharacter() {
            if (charIndex < line.length) {
                lineDiv.textContent += line[charIndex];
                charIndex++;
                setTimeout(typeCharacter, 50); // Typing speed (50ms per character)
            } else {
                lineIndex++;
                if (lineIndex < results.length) {
                    setTimeout(() => typeLine(results[lineIndex]), 500); // Pause before the next line
                }
            }
        }

        typeCharacter();
    }

    // Function to start the typing effect
    function typeEffect() {
        if (results.length > 0) {
            typeLine(results[lineIndex]); // Start typing the first line
        } else {
            outputDiv.textContent = "No results to display."; // Fallback message
        }
    }

    typeEffect();

    // Show loading message on form submission
    const form = document.getElementById("upload-form");
    const loadingMessage = document.getElementById("loading-message");

    if (form) {
        form.addEventListener("submit", function (e) {
            e.preventDefault();
            // Show the loading message when the form is submitted
            showloadingMessage()

            setTimeout(() => {
                form.submit(); // Submit the form programmatically after 2 seconds
            }, 1000); // Delay of 1 seconds
        });
    }
});
