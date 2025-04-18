SELECT * FROM vendas
ORDER BY id_venda ASC

SELECT regiao, categoria, SUM(valor_total) AS total_vendas
FROM vendas
GROUP BY ROLLUP(regiao, categoria)
ORDER BY regiao, categoria;

SELECT 
    canal,
    EXTRACT(MONTH FROM data_venda) AS mes,
    SUM(valor_total) AS total_vendas
FROM vendas
GROUP BY ROLLUP(canal, mes)
ORDER BY canal, mes;

SELECT id_produto, SUM(valor_total) AS receita_total
FROM vendas
GROUP BY id_produto
ORDER BY receita_total DESC
LIMIT 5;

SELECT regiao, canal, SUM(valor_total) AS total_vendas
FROM vendas
GROUP BY CUBE(regiao, canal)
ORDER BY regiao, canal;
