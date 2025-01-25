document.getElementById('storyForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const image = document.getElementById('image').files[0];
    const prompt = document.getElementById('prompt').value;

    formData.append('image', image);
    formData.append('prompt', prompt);

    const output = document.getElementById('output');
    output.innerHTML = `<h3>Generating Story...</h3><p id="storyText"></p>`;

    try {
        const response = await fetch('/generate-story', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            output.innerHTML = `<p style="color: red;">Error: ${errorData.error}</p>`;
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let storyText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            storyText += decoder.decode(value, { stream: true });
            document.getElementById('storyText').textContent = storyText;
        }
    } catch (error) {
        console.error('Error:', error);
        output.innerHTML = `<p style="color: red;">An error occurred while generating the story.</p>`;
    }
});
