const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, SectionType
} = require('/home/gabriel/.nvm/versions/node/v20.20.0/lib/node_modules/docx');
const fs = require('fs');

// ── ABNT NBR 14724 ────────────────────────────────────────────────────────────
// Papel: A4 (11906 x 16838 DXA)
// Margens: Esquerda 3cm, Superior 3cm, Direita 2cm, Inferior 2cm
// Fonte: Times New Roman 12pt
// Espaçamento corpo: 1,5 (360 twips/linha)
// Espaçamento citações/notas/referências: simples (240)
// Recuo parágrafo: 1,25cm (709 DXA)
// Títulos seção 1: NEGRITO MAIÚSCULAS
// Títulos seção 2: Negrito Título (não maiúsculas)
// Numeração de página: canto superior direito, a partir da introdução
// ─────────────────────────────────────────────────────────────────────────────

const marginLeft   = 1701; // 3cm
const marginRight  = 1134; // 2cm
const marginTop    = 1701; // 3cm
const marginBottom = 1134; // 2cm
const pageWidth    = 11906;
const contentWidth = pageWidth - marginLeft - marginRight; // 9071 DXA

const TNR   = "Times New Roman";
const SZ12  = 24;  // 12pt
const SZ10  = 20;  // 10pt (citações longas, legendas de fonte)

// ── Helpers ──────────────────────────────────────────────────────────────────

const emptyLine = () => new Paragraph({
  spacing: { line: 360 },
  children: [new TextRun({ text: "", font: TNR, size: SZ12 })]
});

// Parágrafo de corpo (1,5 espaçamento, recuo 1,25cm, justificado)
const body = (text, opts = {}) => {
  const { indent = true, center = false, bold = false, size = SZ12, spacing = 360 } = opts;
  return new Paragraph({
    alignment: center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    spacing: { line: spacing, before: 0, after: 0 },
    indent: indent && !center ? { firstLine: 709 } : undefined,
    children: [new TextRun({ text, font: TNR, bold, size })]
  });
};

// Título nível 1: NEGRITO MAIÚSCULAS, sem recuo, espaço antes/depois
const h1 = (num, title) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { line: 360, before: 480, after: 240 },
  children: [new TextRun({ text: `${num} ${title.toUpperCase()}`, font: TNR, bold: true, size: SZ12 })]
});

// Título nível 2: Negrito, sem maiúsculas, sem recuo
const h2 = (num, title) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { line: 360, before: 360, after: 120 },
  children: [new TextRun({ text: `${num} ${title}`, font: TNR, bold: true, size: SZ12 })]
});

// Item de lista (• sem uso de unicode problemático — usa hífen ABNT)
const listItem = (text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 360, before: 0, after: 0 },
  indent: { left: 709 },
  children: [new TextRun({ text: `\u2013 ${text}`, font: TNR, size: SZ12 })]
});

// Legenda de tabela (acima): fonte 12, alinhada à esquerda
const tableLabel = (text) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { line: 240, before: 360, after: 60 },
  children: [new TextRun({ text, font: TNR, bold: false, size: SZ12 })]
});

// Fonte abaixo da tabela: fonte 10, alinhada à esquerda
const tableFootnote = (text = "Fonte: Elaboração própria (2026).") => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { line: 240, before: 60, after: 360 },
  children: [new TextRun({ text, font: TNR, size: SZ10 })]
});

// Código (monospace, recuo 4cm = 2268 DXA)
const codeLine = (text) => new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { line: 240, before: 0, after: 0 },
  indent: { left: 2268 },
  children: [new TextRun({ text, font: "Courier New", size: 18 })]
});

// Referência bibliográfica: espaçamento simples, sem recuo (ABNT)
const ref = (text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 240, before: 0, after: 240 },
  children: [new TextRun({ text, font: TNR, size: SZ12 })]
});

// ── Célula de tabela ──────────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };

const tc = (text, opts = {}) => {
  const { bold = false, header = false, align = AlignmentType.LEFT, w, span } = opts;
  return new TableCell({
    borders,
    width: w ? { size: w, type: WidthType.DXA } : undefined,
    shading: header ? { fill: "E8E8E8", type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    columnSpan: span,
    children: [new Paragraph({
      alignment: align,
      spacing: { line: 240 },
      children: [new TextRun({ text, font: TNR, bold: bold || header, size: SZ10 })]
    })]
  });
};

const tr = (...cells) => new TableRow({ children: cells });

// ── TABELAS ───────────────────────────────────────────────────────────────────
const c3 = Math.floor(contentWidth / 3);

const tabela1 = new Table({
  width: { size: contentWidth, type: WidthType.DXA },
  columnWidths: [c3, c3, c3],
  rows: [
    tr(tc("Métrica", {bold:true,header:true,w:c3}), tc("Gradiente Descendente",{bold:true,header:true,w:c3}), tc("Método de Newton",{bold:true,header:true,w:c3})),
    tr(tc("Iterações até convergência",{w:c3}), tc("35",{w:c3}), tc("3",{w:c3})),
    tr(tc("Perda inicial L(θ₀)",{w:c3}), tc("0,69",{w:c3}), tc("0,69",{w:c3})),
    tr(tc("Perda final L(θ*)",{w:c3}), tc("0,3082",{w:c3}), tc("0,3082",{w:c3})),
    tr(tc("||∇L(θ*)|| final",{w:c3}), tc("1,00 × 10⁻⁴",{w:c3}), tc("6,54 × 10⁻⁶",{w:c3})),
    tr(tc("Critério de parada atingido",{w:c3}), tc("Sim (||∇L|| < 10⁻⁴)",{w:c3}), tc("Sim (||∇L|| < 10⁻⁴)",{w:c3})),
    tr(tc("Tempo de execução",{w:c3}), tc("0,54 s",{w:c3}), tc("0,07 s",{w:c3})),
    tr(tc("Convergência teórica",{w:c3}), tc("Linear",{w:c3}), tc("Quadrática",{w:c3})),
  ]
});

const c4 = Math.floor(contentWidth / 4);
const tabela2 = new Table({
  width: { size: contentWidth, type: WidthType.DXA },
  columnWidths: [c4, c4, c4, c4],
  rows: [
    tr(tc("Método de Busca",{bold:true,header:true,w:c4}), tc("Iterações",{bold:true,header:true,w:c4}), tc("Perda Final",{bold:true,header:true,w:c4}), tc("Tempo (s)",{bold:true,header:true,w:c4})),
    tr(tc("Seção Áurea",{w:c4}),              tc("35",{w:c4}), tc("0,3082",{w:c4}), tc("0,54",{w:c4})),
    tr(tc("Partição Igual (Trissecção)",{w:c4}), tc("35",{w:c4}), tc("0,3082",{w:c4}), tc("0,74",{w:c4})),
    tr(tc("Ajuste Quadrático",{w:c4}),        tc("35",{w:c4}), tc("0,3082",{w:c4}), tc("0,19",{w:c4})),
  ]
});

const tabela3 = new Table({
  width: { size: contentWidth, type: WidthType.DXA },
  columnWidths: [c3, c3, c3],
  rows: [
    tr(tc("Métrica",{bold:true,header:true,w:c3}), tc("Sem Balanceamento",{bold:true,header:true,w:c3}), tc("Com Balanceamento",{bold:true,header:true,w:c3})),
    tr(tc("Acurácia",{w:c3}),             tc("86,53%",{w:c3}), tc("81,34%",{w:c3})),
    tr(tc("Precisão",{w:c3}),             tc("68,84%",{w:c3}), tc("45,12%",{w:c3})),
    tr(tc("Recall (Sensibilidade)",{w:c3}), tc("29,23%",{w:c3}), tc("75,38%",{w:c3})),
    tr(tc("F1-Score",{w:c3}),             tc("41,04%",{w:c3}), tc("56,45%",{w:c3})),
    tr(tc("AUC-ROC",{w:c3}),              tc("87,34%",{w:c3}), tc("87,89%",{w:c3})),
  ]
});

// Tabela 4: cabeçalho duplo com rowspan/colspan
const c5 = Math.floor(contentWidth / 5);
const tabela4 = new Table({
  width: { size: contentWidth, type: WidthType.DXA },
  columnWidths: [c5, c5, c5, c5, c5],
  rows: [
    new TableRow({ children: [
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, rowSpan:2, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, verticalAlign:VerticalAlign.CENTER, children:[new Paragraph({alignment:AlignmentType.CENTER, spacing:{line:240}, children:[new TextRun({text:"",font:TNR,bold:true,size:SZ10})]})] }),
      new TableCell({ borders, width:{size:c5*2,type:WidthType.DXA}, columnSpan:2, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER, spacing:{line:240}, children:[new TextRun({text:"Sem Balanceamento",font:TNR,bold:true,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5*2,type:WidthType.DXA}, columnSpan:2, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER, spacing:{line:240}, children:[new TextRun({text:"Com Balanceamento",font:TNR,bold:true,size:SZ10})]})]})
    ]}),
    new TableRow({ children: [
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"Predito: Adimplente",font:TNR,bold:true,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"Predito: Inadimplente",font:TNR,bold:true,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"Predito: Adimplente",font:TNR,bold:true,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"Predito: Inadimplente",font:TNR,bold:true,size:SZ10})]})]})
    ]}),
    new TableRow({ children: [
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"Real: Adimplente",font:TNR,bold:true,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"VN = 1657",font:TNR,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"FP = 43",font:TNR,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"VN = 1401",font:TNR,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"FP = 299",font:TNR,size:SZ10})]})]})
    ]}),
    new TableRow({ children: [
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, shading:{fill:"E8E8E8",type:ShadingType.CLEAR}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"Real: Inadimplente",font:TNR,bold:true,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"FN = 231",font:TNR,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"VP = 95",font:TNR,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"FN = 80",font:TNR,size:SZ10})]})]}),
      new TableCell({ borders, width:{size:c5,type:WidthType.DXA}, margins:{top:80,bottom:80,left:120,right:120}, children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{line:240},children:[new TextRun({text:"VP = 246",font:TNR,size:SZ10})]})]})
    ]}),
  ]
});

// ── HEADER com número de página (canto superior direito) ──────────────────────
const headerComPagina = new Header({
  children: [new Paragraph({
    alignment: AlignmentType.RIGHT,
    children: [new TextRun({ children: [PageNumber.CURRENT], font: TNR, size: SZ12 })]
  })]
});

const headerVazio = new Header({ children: [new Paragraph({ children: [] })] });

// ── CAPA (seção 1: sem número de página) ──────────────────────────────────────
const capaChildren = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360,before:0,after:0}, children:[new TextRun({text:"UNIVERSIDADE FEDERAL DA PARAÍBA",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"CENTRO DE INFORMÁTICA",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"CURSO DE CIÊNCIA DE DADOS E INTELIGÊNCIA ARTIFICIAL",font:TNR,bold:true,size:SZ12})] }),
  emptyLine(), emptyLine(), emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"GABRIEL NEGREIROS SARAIVA",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"JÚLIA MORAES DA SILVA",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"LUIZ EDUARDO DE ALMEIDA SIQUEIRA SILVA",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"PAULO VICTOR CORDEIRO RUFINO DE ARAÚJO",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"PEDRO LUCAS SIMÕES CABRAL",font:TNR,bold:true,size:SZ12})] }),
  emptyLine(), emptyLine(), emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"PREVISÃO DE INADIMPLÊNCIA BANCÁRIA VIA REGRESSÃO LOGÍSTICA:",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"UMA ABORDAGEM DE OTIMIZAÇÃO NÃO LINEAR",font:TNR,bold:true,size:SZ12})] }),
  emptyLine(), emptyLine(), emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"RELATÓRIO FINAL",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"Disciplina: Otimização Não Linear",font:TNR,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"Professor: Felipe A. G. Moreno",font:TNR,size:SZ12})] }),
  emptyLine(), emptyLine(), emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"JOÃO PESSOA",font:TNR,bold:true,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"2026",font:TNR,bold:true,size:SZ12})] }),
];

// ── SUMÁRIO (seção 2: sem número de página) ───────────────────────────────────
const sumarioEntradas = [
  ["1 INTRODUÇÃO", "3"],
  ["2 DESENVOLVIMENTO DO PROBLEMA ESPECÍFICO APLICADO", "3"],
  ["2.1 Caracterização do Problema", "3"],
  ["2.2 Base de Dados", "4"],
  ["2.3 Formulação Matemática", "4"],
  ["2.4 Gradiente e Hessiana", "5"],
  ["2.5 Prova de Convexidade", "5"],
  ["3 RESOLUÇÃO DO PROBLEMA", "6"],
  ["3.1 Estratégia de Resolução", "6"],
  ["3.2 Métodos de Busca em Linha", "6"],
  ["3.3 Gradiente Descendente com Busca em Linha", "7"],
  ["3.4 Método de Newton com Busca em Linha", "7"],
  ["4 METODOLOGIA", "8"],
  ["4.1 Abordagem Adotada", "8"],
  ["4.2 Configuração dos Experimentos", "8"],
  ["4.3 Referências Bibliográficas Metodológicas", "9"],
  ["5 IMPLEMENTAÇÃO DOS ALGORITMOS", "9"],
  ["5.1 Estrutura da Implementação", "9"],
  ["5.2 Classe RegressaoLogistica", "9"],
  ["5.3 Exemplo de Uso", "10"],
  ["6 RESULTADOS E CONCLUSÕES", "10"],
  ["6.1 Comparação dos Métodos de Otimização", "10"],
  ["6.2 Comparação dos Métodos de Busca em Linha", "11"],
  ["6.3 Análise do Desbalanceamento de Classes", "11"],
  ["6.4 Conclusões", "13"],
  ["REFERÊNCIAS", "15"],
];

const sumarioChildren = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: 360, before: 0, after: 480 },
    children: [new TextRun({ text: "SUMÁRIO", font: TNR, bold: true, size: SZ12 })]
  }),
  ...sumarioEntradas.map(([titulo, pagina]) => {
    const pontos = Math.max(3, 65 - titulo.length - pagina.length);
    const isSub = /^\d+\.\d/.test(titulo);
    return new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: 360, before: 0, after: 0 },
      indent: isSub ? { left: 360 } : undefined,
      children: [new TextRun({ text: `${titulo} ${"·".repeat(pontos)} ${pagina}`, font: TNR, size: SZ12 })]
    });
  })
];

// ── CONTEÚDO PRINCIPAL (seção 3: com número de página) ────────────────────────
const conteudoChildren = [
  // 1. INTRODUÇÃO
  h1("1", "Introdução"),
  body("A inadimplência bancária representa um dos principais riscos operacionais enfrentados por instituições financeiras, impactando diretamente a rentabilidade, a necessidade de provisões de crédito e a estabilidade do sistema financeiro. A capacidade de prever, com antecedência, a probabilidade de um cliente tornar-se inadimplente é de alto valor estratégico, tanto para a gestão de risco quanto para a concessão responsável de crédito."),
  emptyLine(),
  body("No contexto do aprendizado de máquina supervisionado, o treinamento de modelos preditivos pode ser formulado como um problema de otimização: busca-se determinar os parâmetros de um modelo que minimizem uma função de perda sobre os dados de treinamento. No caso da Regressão Logística — modelo amplamente utilizado em credit scoring —, esse problema toma a forma de um problema de otimização não linear irrestrito, com função objetiva convexa e diferenciável, o que permite a aplicação de métodos com garantias formais de convergência."),
  emptyLine(),
  body("O presente trabalho é desenvolvido no âmbito da disciplina Otimização Não Linear do Curso de Ciência de Dados e Inteligência Artificial da Universidade Federal da Paraíba (UFPB), sob orientação do professor Felipe A. G. Moreno. O objetivo é formular o treinamento da Regressão Logística como um problema de otimização não linear, implementar dois métodos de solução — Gradiente Descendente e Método de Newton, ambos com busca em linha exata — e compará-los empiricamente em termos de velocidade de convergência, eficiência computacional e qualidade da solução obtida."),
  emptyLine(),
  body("A base de dados utilizada é o dataset de inadimplência de clientes do Banco, composto por 10.127 registros e 14 atributos preditivos. O trabalho abrange desde a formulação matemática do problema, passando pela prova formal de convexidade da função de perda, até a análise dos resultados experimentais obtidos."),

  // 2. DESENVOLVIMENTO
  h1("2", "Desenvolvimento do Problema Específico Aplicado"),
  h2("2.1", "Caracterização do Problema"),
  body("O problema enquadra-se na classe de Problemas de Otimização em Aprendizado Supervisionado — Classificação Binária. No aprendizado supervisionado, o modelo é treinado a partir de um conjunto de pares (xᵢ, yᵢ), onde xᵢ ∈ ℝᵖ é o vetor de atributos do i-ésimo cliente e yᵢ ∈ {0, 1} é o rótulo binário (0 = adimplente, 1 = inadimplente). O objetivo é aprender uma função ƒ: ℝᵖ → [0, 1] que estime a probabilidade de inadimplência, minimizando um critério de erro sobre os dados de treinamento."),
  emptyLine(),
  h2("2.2", "Base de Dados"),
  body("Foi utilizado o dataset de inadimplência de crédito disponibilizado, contendo informações financeiras e comportamentais de clientes bancários. As características da base de dados são:"),
  emptyLine(),
  listItem("Total de registros: 10.127 clientes"),
  listItem("Atributos preditivos (após codificação): 23 features"),
  listItem("Variável resposta: default (0 = adimplente, 1 = inadimplente)"),
  listItem("Divisão treino/teste: 80%/20%, semente aleatória = 42"),
  emptyLine(),
  body("As variáveis preditivas incluem: idade, sexo, número de dependentes, escolaridade, estado civil, faixa de salário anual, tipo de cartão, meses de relacionamento com o banco, quantidade de produtos contratados, número de interações nos últimos 12 meses, meses de inatividade, limite de crédito, valor total e quantidade de transações nos últimos 12 meses. As variáveis categóricas foram convertidas em códigos numéricos, e todas as features foram normalizadas pelo escore-z: x̃ = (x − μ) / σ, garantindo que os atributos estejam na mesma escala antes da otimização."),
  emptyLine(),
  h2("2.3", "Formulação Matemática"),
  body("O modelo de Regressão Logística estima a probabilidade de inadimplência pela função sigmoide aplicada à combinação linear dos atributos:"),
  emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"P(y = 1 | x) = σ(xᵀθ) = 1 / (1 + e^(−xᵀθ))",font:TNR,size:SZ12})] }),
  emptyLine(),
  body("onde θ ∈ ℝᵖ⁺¹ é o vetor de parâmetros a ser determinado e x ∈ ℝᵖ⁺¹ é o vetor de atributos com coluna de 1 adicionada. O treinamento é formulado como o seguinte problema de minimização não linear:"),
  emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"min   L(θ) = −(1/n) Σᵢ [yᵢ log σ(xᵢᵀθ) + (1 − yᵢ) log(1 − σ(xᵢᵀθ))] + (λ/2)||θ₁:||²",font:TNR,size:SZ12})] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"θ ∈ ℝᵖ⁺¹",font:TNR,size:SZ12})] }),
  emptyLine(),
  body("onde o primeiro termo é a entropia cruzada binária (verossimilhança negativa) e o segundo é a regularização L2 com parâmetro λ, que penaliza parâmetros de grande magnitude para evitar sobreajuste. Por convenção, o viés θ₀ não é regularizado."),
  emptyLine(),
  h2("2.4", "Gradiente e Hessiana"),
  body("O gradiente da função de perda, necessário para os métodos de otimização, é dado por:"),
  emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"∇L(θ) = (1/n) Xᵀ(σ(Xθ) − y) + λ [0, θ₁, θ₂, ..., θₚ]ᵀ",font:TNR,size:SZ12})] }),
  emptyLine(),
  body("A Hessiana — necessária para o Método de Newton — é:"),
  emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"H(θ) = (1/n) Xᵀ W X + λ diag(0, 1, 1, ..., 1)",font:TNR,size:SZ12})] }),
  emptyLine(),
  body("onde W = diag(wᵢ) com wᵢ = σ(xᵢᵀθ)(1 − σ(xᵢᵀθ)) é a matriz diagonal de pesos de incerteza, cujos elementos satisfazem 0 < wᵢ ≤ 1/4 para todo θ, pois σ(z) ∈ (0, 1) para z ∈ ℝ."),
  emptyLine(),
  h2("2.5", "Prova de Convexidade"),
  body("A convexidade da função L(θ) é garantida pela semi-definição positiva da Hessiana. Para qualquer vetor v ∈ ℝᵖ⁺¹:"),
  emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"vᵀ H(θ) v = (1/n)(Xv)ᵀ W (Xv) = (1/n) Σᵢ wᵢ (xᵢᵀv)² ≥ 0",font:TNR,size:SZ12})] }),
  emptyLine(),
  body("Como wᵢ > 0, o primeiro termo é não negativo, e a regularização L2 adiciona um termo estritamente positivo nas direções associadas aos coeficientes (exceto o viés). Para λ > 0, a função torna-se estritamente convexa em condições usuais, implicando que: (i) qualquer mínimo local é global; (ii) o mínimo global é único; (iii) o Método de Newton apresenta convergência quadrática local. Empiricamente, isso se reflete em autovalores não negativos da Hessiana ao longo da otimização."),

  // 3. RESOLUÇÃO
  h1("3", "Resolução do Problema"),
  h2("3.1", "Estratégia de Resolução"),
  body("O problema de minimização irrestrito de L(θ) é resolvido por dois métodos iterativos de descida, ambos combinados com busca em linha exata para determinação do passo ótimo α*. Em cada iteração k, dado o ponto atual θₖ e uma direção de descida dₖ, determina-se o passo ótimo:"),
  emptyLine(),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing:{line:360}, children:[new TextRun({text:"α* = argmin_α L(θₖ + α dₖ),   α > 0",font:TNR,size:SZ12})] }),
  emptyLine(),
  body("O intervalo inicial [a, b] contendo o mínimo univariado é determinado automaticamente por um algoritmo de duplicação de passo a partir de α = 0, que expande progressivamente até detectar que g(α) = L(θₖ + α dₖ) começa a crescer."),
  emptyLine(),
  h2("3.2", "Métodos de Busca em Linha"),
  body("Três métodos de busca em linha foram implementados para resolver o problema univariado:"),
  emptyLine(),
  body("a) Seção Áurea: utiliza a razão áurea φ = (√5 − 1)/2 ≈ 0,618 para particionar o intervalo, garantindo apenas 1 avaliação nova por iteração. Reduz o intervalo por um fator φ a cada passo, sendo ótimo para funções unimodais.", {indent:false}),
  emptyLine(),
  body("b) Partição Igual (Trissecção): divide o intervalo [a, b] em três partes iguais e descarta o terço externo com maior valor. Requer 2 avaliações por iteração e reduz o intervalo por 2/3. Mais simples, porém menos eficiente que a Seção Áurea.", {indent:false}),
  emptyLine(),
  body("c) Ajuste Quadrático: ajusta uma parábola por três pontos e usa seu vértice como nova estimativa do mínimo. Substitui o pior dos três pontos a cada iteração. Converge mais rapidamente para funções suaves com curvatura bem definida.", {indent:false}),
  emptyLine(),
  h2("3.3", "Gradiente Descendente com Busca em Linha"),
  body("O Gradiente Descendente (GD) utiliza o negativo do gradiente como direção de descida, que é a direção de maior decrescimento local de L(θ). O algoritmo executa as seguintes etapas:"),
  emptyLine(),
  ...["1.  Inicializa θ₀ = 0 ∈ ℝᵖ⁺¹","2.  Calcula o gradiente ∇L(θₖ)","3.  Se ||∇L(θₖ)|| < ε (com ε = 10⁻⁴), interrompe: solução encontrada","4.  Define a direção dₖ = −∇L(θₖ)","5.  Determina o passo ótimo: α* = argmin_α L(θₖ + α dₖ) via busca em linha","6.  Atualiza: θₖ₊₁ = θₖ + α* dₖ","7.  Repete a partir do passo 2"].map(listItem),
  emptyLine(),
  body("A convergência do GD é linear: o erro decresce, a cada iteração, por um fator proporcional a (κ(H) − 1)/(κ(H) + 1), onde κ(H) = λₘₐₓ/λₘᵢₙ é o número de condição da Hessiana. Quanto maior o número de condição, mais lenta a convergência."),
  emptyLine(),
  h2("3.4", "Método de Newton com Busca em Linha"),
  body("O Método de Newton utiliza a informação de curvatura da Hessiana para escalar a direção de descida, eliminando a dependência do número de condições. O algoritmo:"),
  emptyLine(),
  ...["1.  Inicializa θ₀ = 0 ∈ ℝᵖ⁺¹","2.  Calcula ∇L(θₖ) e H(θₖ)","3.  Se ||∇L(θₖ)|| < ε, interrompe","4.  Resolve o sistema linear (H(θₖ) + εI) dₖ = −∇L(θₖ), obtendo a direção de Newton","5.  Determina o passo ótimo: α* = argmin_α L(θₖ + α dₖ) via busca em linha","6.  Atualiza: θₖ₊₁ = θₖ + α* dₖ","7.  Repete a partir do passo 2"].map(listItem),
  emptyLine(),
  body("A regularização de Tikhonov (H + εI, com ε = 10⁻⁸) garante estabilidade numérica para casos em que H se aproxima da singularidade. A convergência do Método de Newton é quadrática nas vizinhanças do ponto ótimo — o número de dígitos corretos aproximadamente dobra a cada iteração — tornando-o dramaticamente mais eficiente que o GD em problemas com condicionamento desfavorável."),

  // 4. METODOLOGIA
  h1("4", "Metodologia"),
  h2("4.1", "Abordagem Adotada"),
  body("Conforme discutido no relatório de avanço do projeto, embora métodos baseados em árvores e ensemble (como Random Forest) apresentem alta capacidade preditiva, optou-se pela Regressão Logística por três razões principais: (i) sua fundamentação estatística sólida e interpretabilidade dos coeficientes; (ii) a formulação explícita como problema de otimização não linear contínuo, alinhada aos objetivos da disciplina; e (iii) a possibilidade de análise formal das propriedades de convexidade, existência e unicidade do mínimo."),
  emptyLine(),
  body("A estimação dos parâmetros é realizada via minimização da entropia cruzada binária com regularização L2. Dois algoritmos de otimização são comparados — Gradiente Descendente e Método de Newton —, cada um testado com três métodos de busca em linha: Seção Áurea, Partição Igual e Ajuste Quadrático."),
  emptyLine(),
  h2("4.2", "Configuração dos Experimentos"),
  body("Os hiperparâmetros adotados em todos os experimentos são:"),
  emptyLine(),
  listItem("Regularização L2: λ = 0,01"),
  listItem("Critério de parada: ||∇L(θ)|| < 10⁻⁴"),
  listItem("Máximo de iterações: 500 (GD) e 100 (Newton)"),
  listItem("Inicialização: θ₀ = 0"),
  listItem("Normalização: escore-z (μ = 0, σ = 1 por feature)"),
  listItem("Divisão treino/teste: 80%/20%, semente = 42"),
  emptyLine(),
  h2("4.3", "Referências Bibliográficas Metodológicas"),
  body("A metodologia baseia-se nos trabalhos de referência da área: SUN et al. (2019) apresentam um panorama dos métodos de otimização em aprendizado de máquina, com ênfase nas propriedades de convergência de algoritmos de gradiente e métodos de segunda ordem. GAMBELLA, GHADDAR e NAOUM-SAWAYA (2021) categorizam os problemas de otimização em aprendizado de máquina segundo a classe do modelo e a estrutura matemática da função objetivo. Do ponto de vista aplicado, CAETANO (2024) e SOUZA (2025) demonstram a eficácia da Regressão Logística em contextos de modelagem de risco de crédito e previsão de inadimplência."),

  // 5. IMPLEMENTAÇÃO
  h1("5", "Implementação dos Algoritmos"),
  h2("5.1", "Estrutura da Implementação"),
  body("A implementação foi realizada integralmente em Python, sem o uso de bibliotecas de aprendizado de máquina para os algoritmos de otimização. Todo o código encontra-se comentado e disponível no Jupyter Notebook principal (main.ipynb) e no módulo regressao_log.py, ambos no repositório do projeto."),
  emptyLine(),
  body("O módulo regressao_log.py implementa a classe RegressaoLogistica, que encapsula todos os algoritmos de forma modular e reutilizável. A interface pública segue o padrão do scikit-learn, com os métodos fit(), predict() e predict_prob()."),
  emptyLine(),
  h2("5.2", "Classe RegressaoLogistica"),
  body("A classe aceita os seguintes parâmetros de configuração:"),
  emptyLine(),
  listItem('metodo_otimizacao: "gradiente_descendente" ou "newton"'),
  listItem('metodo_busca: "secao_aurea", "particao_igual" ou "ajuste_quadratico"'),
  listItem("tmax: número máximo de iterações (padrão: 1.000)"),
  listItem("tolerancia: critério de parada ||∇L(θ)|| < tolerancia (padrão: 10⁻⁶)"),
  listItem("lambda_: coeficiente de regularização L2 (padrão: 0,01)"),
  emptyLine(),
  body("Após o treinamento via fit(X, y), os atributos historico_perda_, historico_norma_grad_, historico_alpha_, historico_tempo_ e n_iteracoes_ registram a trajetória de convergência para análise posterior."),
  emptyLine(),
  h2("5.3", "Exemplo de Uso"),
  emptyLine(),
  codeLine("from regressao_log import RegressaoLogistica"),
  codeLine("from sklearn.preprocessing import StandardScaler"),
  codeLine(""),
  codeLine("scaler = StandardScaler()"),
  codeLine("X_train_scaled = scaler.fit_transform(X_train)"),
  codeLine("X_test_scaled  = scaler.transform(X_test)"),
  codeLine(""),
  codeLine('modelo = RegressaoLogistica(metodo_otimizacao="newton",'),
  codeLine('    metodo_busca="secao_aurea", tmax=50, tolerancia=1e-4, lambda_=0.01)'),
  codeLine("modelo.fit(X_train_scaled, y_train)"),
  codeLine("y_pred = modelo.predict(X_test_scaled)"),
  emptyLine(),

  // 6. RESULTADOS
  h1("6", "Resultados e Conclusões"),
  h2("6.1", "Comparação dos Métodos de Otimização"),
  body("A Tabela 1 apresenta os resultados comparativos dos dois métodos de otimização, ambos configurados com busca em linha por Seção Áurea, critério de parada ‖∇L‖ < 10⁻⁴, regularização λ = 0,01 e limite de 500 iterações para o GD. O número de features expandiu-se para 23 após a codificação one-hot das variáveis categóricas."),
  emptyLine(),
  tableLabel("Tabela 1 – Comparação entre Gradiente Descendente e Método de Newton"),
  tabela1,
  tableFootnote(),
  body("Os resultados evidenciam a superioridade do Método de Newton em termos de eficiência: convergiu para o mínimo em apenas 3 iterações, enquanto o Gradiente Descendente precisou de 35 iterações para satisfazer o critério de parada. Ambos alcançaram a mesma perda final (L(θ*) ≈ 0,3082), confirmando convergência para a mesma solução — esperado dado que a função é estritamente convexa com mínimo global único."),
  emptyLine(),
  body("Com a normalização z-score aplicada, o número de condição da Hessiana manteve-se em κ ≈ 12 ao longo da otimização. O Método de Newton, ao incorporar a informação de curvatura via Hessiana, é teoricamente imune ao número de condição e exibe convergência quadrática local, o que explica a convergência em apenas 3 iterações, com tempo de execução 7,7 vezes menor que o GD."),
  emptyLine(),
  h2("6.2", "Comparação dos Métodos de Busca em Linha"),
  body("A Tabela 2 apresenta os resultados da comparação entre os três métodos de busca em linha, aplicados ao Gradiente Descendente com limite de 200 iterações."),
  emptyLine(),
  tableLabel("Tabela 2 – Comparação dos Métodos de Busca em Linha (Gradiente Descendente)"),
  tabela2,
  tableFootnote(),
  body("Com a normalização z-score, todos os três métodos de busca convergiram em 35 iterações com a mesma perda final (0,3082), demonstrando que a função de perda bem condicionada torna os métodos equivalentes em qualidade de solução. A diferença observou-se no tempo de execução: o Ajuste Quadrático foi o mais eficiente (0,19 s), por requerer menos avaliações de g(α) em funções suaves. A Seção Áurea apresentou desempenho intermediário (0,54 s), enquanto a Partição Igual foi a mais lenta (0,74 s) por requerer 2 avaliações por iteração."),
  emptyLine(),
  h2("6.3", "Análise do Desbalanceamento de Classes"),
  body("O dataset apresenta apenas 16,1% de inadimplentes (classe positiva), o que torna a classificação naturalmente enviesada para a classe majoritária. Para investigar o impacto desse desequilíbrio, foram conduzidos experimentos paralelos com e sem balanceamento no conjunto de treino, mediante a técnica SMOTE (Synthetic Minority Over-sampling Technique), que gera amostras sintéticas da classe minoritária por interpolação entre exemplos reais vizinhos."),
  emptyLine(),
  tableLabel("Tabela 3 – Métricas de Classificação no Conjunto de Teste (2.026 clientes)"),
  tabela3,
  tableFootnote(),
  body("O modelo sem balanceamento apresenta acurácia de 86,53% e precisão de 68,84%, mas recall baixo (29,23%): a maioria dos inadimplentes reais não é detectada. Com o balanceamento SMOTE no treino, o recall sobe para 75,38% — ganho de aproximadamente 46 pontos percentuais — ao custo de queda na precisão (45,12%) e na acurácia (81,34%). O AUC-ROC permanece elevado em ambos os casos (87,34% e 87,89%), indicando que a capacidade discriminativa global do modelo é robusta independentemente do balanceamento."),
  emptyLine(),
  body("O F1-Score, que pondera precisão e recall, revela o modelo balanceado como superior: 56,45% versus 41,04%. Em aplicações financeiras de gestão de risco, onde o custo de conceder crédito a um inadimplente (falso negativo) tipicamente supera o custo de recusar crédito a um bom pagador (falso positivo), a configuração balanceada é a mais adequada."),
  emptyLine(),
  tableLabel("Tabela 4 – Matriz de Confusão (limiar de classificação = 0,5)"),
  tabela4,
  tableFootnote(),
  body("Adicionalmente, a otimização do limiar de decisão no modelo balanceado identificou o limiar ótimo de 0,53 (em vez do padrão 0,5), que maximiza o F1-Score para 0,5677, com recall de 72,92% e precisão de 46,47%. A análise das distribuições de probabilidade predita reforça esse resultado: sem balanceamento, a mediana de P(y=1|x) para inadimplentes é 0,3302; com balanceamento, sobe para 0,7318 — indicando que o modelo balanceado atribui probabilidades mais discriminativas à classe positiva."),
  emptyLine(),
  h2("6.4", "Conclusões"),
  body("Este trabalho demonstrou que o problema de treinamento da Regressão Logística pode ser rigorosamente formulado como um problema de otimização não linear irrestrito, com função objetivo estritamente convexa — garantindo existência e unicidade do mínimo global — e diferenciável — permitindo a aplicação de métodos baseados em gradiente com garantias formais de convergência."),
  emptyLine(),
  body("A comparação empírica entre Gradiente Descendente e Método de Newton confirmou as previsões teóricas. Com a normalização z-score garantindo bom condicionamento (κ ≈ 12), o GD convergiu em 35 iterações. O Método de Newton, imune ao número de condições por explorar a curvatura via Hessiana, convergiu em apenas 3 iterações — 11,7 vezes menos iterações e 7,7 vezes menos tempo de execução."),
  emptyLine(),
  body("Na comparação dos métodos de busca em linha, todos os três produziram soluções equivalentes em qualidade e número de iterações em dados normalizados; a diferença observou-se no tempo de execução, com o Ajuste Quadrático sendo o mais eficiente para esta função suave."),
  emptyLine(),
  body("Do ponto de vista preditivo, o modelo com balanceamento de classes é o mais adequado para aplicações de gestão de risco de crédito, atingindo recall de 75,38% e F1-Score de 56,45% — substancialmente superiores ao modelo sem balanceamento (recall 29,23%, F1 41,04%). A otimização do limiar de decisão para 0,53 eleva o recall para 72,92%, demonstrando que estratégias de pós-processamento complementam as técnicas de balanceamento na detecção da classe minoritária."),
  emptyLine(),
  body("Conclui-se que a Regressão Logística, além de ser um modelo de credit scoring eficaz e interpretável, constitui um excelente veículo pedagógico para a aplicação de conceitos centrais de otimização não linear: convexidade, condição de otimalidade de primeira e segunda ordem, métodos de gradiente, métodos de Newton, técnicas de busca em linha e análise de condicionamento de sistemas lineares."),
];

// ── REFERÊNCIAS (seção 4: sem quebra de página automática — nova página manual) ─
const refsChildren = [
  new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing:{line:360,before:0,after:720},
    children:[new TextRun({text:"REFERÊNCIAS",font:TNR,bold:true,size:SZ12})] }),
  ref("CAETANO, Tatiane Moreira. Algoritmos de aprendizado de máquina no estudo da inadimplência em uma instituição financeira. 2024. Trabalho de Conclusão de Curso (Graduação em Estatística) — Universidade Federal de Uberlândia, Uberlândia, 2024."),
  ref("GAMBELLA, Claudio; GHADDAR, Bissan; NAOUM-SAWAYA, Joe. Optimization problems for machine learning: A survey. European Journal of Operational Research, v. 290, n. 3, p. 807-828, 2021."),
  ref("SOUZA, Flávia Braga de. Gestão do risco de crédito com score dinâmico para previsão de inadimplência de pequenas e médias empresas (PME). 2025. Dissertação de Mestrado (Mestre Profissional em Ciências Contábeis) — Pontifícia Universidade Católica de São Paulo, São Paulo, 2025."),
  ref("SUN, Shiliang; CAO, Zehui; ZHU, Han; ZHAO, Jing. A survey of optimization methods from a machine learning perspective. IEEE Transactions on Cybernetics, v. 50, n. 8, p. 3668-3681, 2019."),
];

// ── MONTAGEM DO DOCUMENTO (3 seções) ─────────────────────────────────────────
const pageProps = {
  size: { width: 11906, height: 16838 },
  margin: { top: marginTop, right: marginRight, bottom: marginBottom, left: marginLeft }
};

const doc = new Document({
  styles: {
    default: { document: { run: { font: TNR, size: SZ12 } } }
  },
  sections: [
    // Seção 1 — Capa (sem número de página)
    {
      properties: { page: pageProps, type: SectionType.NEXT_PAGE },
      headers: { default: headerVazio },
      footers: { default: new Footer({ children: [] }) },
      children: capaChildren
    },
    // Seção 2 — Sumário (sem número de página)
    {
      properties: { page: pageProps, type: SectionType.NEXT_PAGE },
      headers: { default: headerVazio },
      footers: { default: new Footer({ children: [] }) },
      children: sumarioChildren
    },
    // Seção 3 — Conteúdo + Referências (com número de página no canto superior direito)
    {
      properties: { page: pageProps, type: SectionType.NEXT_PAGE },
      headers: { default: headerComPagina },
      footers: { default: new Footer({ children: [] }) },
      children: [...conteudoChildren, ...refsChildren]
    }
  ]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Relatorio_Final_ONL_v2.docx", buf);
  console.log("OK — Relatorio_Final_ONL_v2.docx (ABNT) gerado com sucesso");
}).catch(e => { console.error(e); process.exit(1); });
