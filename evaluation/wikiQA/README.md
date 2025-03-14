
# WikiQA Dataset

![Dataset Creation](dataset-creation.png)

## Selected Documents

### Mathemathics 
*   [Prime Numbers](https://en.wikipedia.org/wiki/Prime_number)
*   [Linear Algebra](https://en.wikipedia.org/wiki/Linear_algebra)
*   [Calculus](https://en.wikipedia.org/wiki/Calculus)
*   [Probability](https://en.wikipedia.org/wiki/Probability)

### Computer Science

*   [Algorithm](https://en.wikipedia.org/wiki/Algorithm)
*   [Data structure](https://en.wikipedia.org/wiki/Data_structure)
*   [Artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence)
*   [Computer programming](https://en.wikipedia.org/wiki/Computer_programming)

### Biology

*   [Cell (biology)](https://en.wikipedia.org/wiki/Cell_(biology))
*   [Genetics](https://en.wikipedia.org/wiki/Genetics)
*   [Evolution](https://en.wikipedia.org/wiki/Evolution)
*   [Ecology](https://en.wikipedia.org/wiki/Ecology)

### Physics

*   [Classical mechanics](https://en.wikipedia.org/wiki/Classical_mechanics)
*   [Electromagnetism](https://en.wikipedia.org/wiki/Electromagnetism)
*   [Quantum mechanics](https://en.wikipedia.org/wiki/Quantum_mechanics)
*   [Thermodynamics](https://en.wikipedia.org/wiki/Thermodynamics)

### General - Batman

*   [Batman](https://en.wikipedia.org/wiki/Batman)
*   [Dachshund](https://en.wikipedia.org/wiki/Dachshund)  
*   [Conspiracy theory](https://en.wikipedia.org/wiki/Conspiracy_theory)  
*   [Religion](https://en.wikipedia.org/wiki/Religion)  



## Types of Questions

(✔) Are Included

[✔] 1. Factual Questions:

These questions seek specific, objective answers based on scientific facts or data.

    What is the chemical symbol for water?

    What is the speed of light in a vacuum?

    How many chromosomes do humans have?

[✔] 2. Multihop Questions:

These questions require combining multiple pieces of scientific information or performing sequential reasoning.

    If the half-life of Carbon-14 is 5,730 years, and a fossil has 25% of its original Carbon-14 remaining, how old is the fossil?

    Given that the density of water is 1 g/cm³ and the volume of a container is 500 cm³, what is the mass of the water it can hold?

3. Semantic Questions:

These questions involve interpreting the meaning or significance of scientific concepts or phenomena.

    What does the term "entropy" mean in thermodynamics, and how does it relate to disorder?

    How does the concept of "natural selection" explain the evolution of species over time?

4. Logical Reasoning Questions:

These questions require applying logical principles or scientific laws to solve problems.

    If a force of 10 N is applied to a 2 kg object, what is its acceleration according to Newton's second law?

    If a plant requires 6 hours of sunlight to produce 10 grams of glucose, how much glucose will it produce in 12 hours under the same conditions?

5. Creative Thinking Questions:

These questions require innovative or imaginative thinking to address scientific challenges or scenarios.

    How might you design an experiment to test the effects of microgravity on plant growth?

    What would be a potential solution to reduce carbon emissions in urban areas using renewable energy technologies?

6. Problem-Solving Questions:

These questions involve applying scientific formulas, principles, or methods to solve specific problems.

    Calculate the pH of a solution with a hydrogen ion concentration of 1 x 10⁻⁵ M.

    A ball is dropped from a height of 20 meters. How long will it take to hit the ground, assuming no air resistance?

7. Ethical and Philosophical Questions:

These questions explore the moral, ethical, or philosophical implications of scientific advancements or discoveries.

    Is it ethical to use CRISPR technology to genetically modify human embryos?

    Should scientists be held responsible for the potential misuse of their discoveries, such as nuclear weapons?


## Documents Structure

```py
class Chunk(TypedDict):
    level: int  # heading level (h2 = 0)
    title: str
    text: str
    tokens: int

class Document(TypedDict):
    title: str
    chunks: list[Chunk]
    categories: list[str]

class WikiDocument(TypedDict):
    url: str
    en: Document
    es: Document
```

**Example:** Wikipedia Primes Article

```json
{
    "url": "https://en.wikipedia.org/wiki/Prime_number",
    "en": {
        "title": "Prime number",
        "chunks": [
            {
                "level": 0,
                "title": "Definition and examples",
                "text": "A natural number (1, 2, 3, 4, 5, 6, etc.) is called a prime number (or a prime) ...",
                "tokens": 838
            },
            ...
        ],
        "categories": [ "Category:Prime numbers", ...]
    },
    "es": {
        "title": "Número primo",
        "chunks": [
            {
                "level": 0,
                "title": "El número 1 no se considera primo",
                "text": "La cuestión acerca de si el número 1 debe o no considerarse primo está ...​",
                "tokens": 367
            },
        ],
        "categories": ["Categoría:Números primos", ...]
    }
},
```

## QA Structure 

```py 
class TestCase(TypeDict):
    id: int
    type: Literal["factual", "multihop", "complex"]
```

```json 
{
    "id": 0,
    "type": "factual",
    "question": "What is the largest known prime number as of 2024?",
    "answer": "According to the most recent information available up until October 2024, the largest known prime number is also a Mersenne prime consisting of 41,024,320 decimal",
    "chunks": [ 3 ]
}
```