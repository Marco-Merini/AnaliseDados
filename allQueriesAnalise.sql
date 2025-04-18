SELECT * FROM aula
SELECT * FROM Aulas_assistidas
SELECT * FROM estudante
SELECT * FROM instrutor

SELECT E.nome, A.instituicao, AA.notas
FROM Aulas_assistidas AA
JOIN Estudante E ON AA.estudanteID = E.estudanteID
JOIN Instrutor I ON AA.instrutorID = I.instrutorID
JOIN Aula A ON AA.aulaID = A.aulaID
WHERE A.estado = 'Santa Catarina'
  AND E.curso <> I.curso
  AND AA.notas > 70;

SELECT E.nome, I.instrutorID, AVG(AA.notas) AS media
FROM Aulas_assistidas AA
JOIN Estudante E ON AA.estudanteID = E.estudanteID
JOIN Instrutor I ON AA.instrutorID = I.instrutorID
JOIN Aula A ON AA.aulaID = A.aulaID
WHERE A.cidade = 'Joinville'
GROUP BY E.nome, I.instrutorID;

SELECT I.instrutorID, AVG(AA.notas) AS media
FROM Aulas_assistidas AA
JOIN Estudante E ON AA.estudanteID = E.estudanteID
JOIN Instrutor I ON AA.instrutorID = I.instrutorID
JOIN Aula A ON AA.aulaID = A.aulaID
WHERE A.cidade = 'Joinville'
GROUP BY ROLLUP (I.instrutorID);

SELECT E.curso, AVG(AA.notas) AS media
FROM Aulas_assistidas AA
JOIN Estudante E ON AA.estudanteID = E.estudanteID
GROUP BY E.curso;

SELECT E.curso AS curso_estudante, I.curso AS curso_instrutor, AVG(AA.notas) AS media
FROM Aulas_assistidas AA
JOIN Estudante E ON AA.estudanteID = E.estudanteID
JOIN Instrutor I ON AA.instrutorID = I.instrutorID
GROUP BY E.curso, I.curso;

SELECT A.estado, A.cidade, A.instituicao, AVG(AA.notas) AS media
FROM Aulas_assistidas AA
JOIN Aula A ON AA.aulaID = A.aulaID
GROUP BY ROLLUP (A.estado, A.cidade, A.instituicao);

SELECT A.estado, A.cidade, A.instituicao, AVG(AA.notas) AS media
FROM Aulas_assistidas AA
JOIN Aula A ON AA.aulaID = A.aulaID
GROUP BY CUBE (A.estado, A.cidade, A.instituicao);