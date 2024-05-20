class Source:
    def __init__(self, source):
        self._source = source
        self._i = 0

    def current(self):
        return self._source[self._i]

    def advance(self, k=None):
        if k is None:
            k = 1
        res = self.lookahead(k)
        self._i += k
        return res

    def chars_left(self):
        return len(self._source) - self._i

    def lookahead(self, k=None):
        if k is None:
            k = 1
        k = min(len(self._source), self._i + k)
        return self._source[self._i:k]

class Lexer:
    SHORT1 = {
        '(': 'LPAR',
        ')': 'RPAR',
        '{': 'LBRACE',
        '}': 'RBRACE',
        '[': 'LBRACKET',
        ']': 'RBRACKET',
        ',': 'COMMA',
        '.': 'PERIOD',
        '=': 'ASSIGN',
        '<': 'LESS',
        '>': 'GREATER',
        ';': 'SEMI',
        '?': 'TERN1',
        ':': 'TERN2',
        '+': 'ADD',
        '-': 'SUB',
        '*': 'MULT',
        '/': 'DIV',
        '%': 'MOD',
        '|': 'BOR',
        '^': 'XOR',
        '~': 'BNOT',
    }
    SHORT2 = {
        '==': 'EQUALS',
        '!=': 'NEQUALS',
        '<=': 'LEQ',
        '>=': 'GEQ',
        '+=': 'ADD_ASSIGN',
        '-=': 'SUB_ASSIGN',
        '*=': 'MULT_ASSIGN',
        '/=': 'DIV_ASSIGN',
        '%=': 'MOD_ASSIGN',
        '&=': 'BAND_ASSIGN',
        '|=': 'BOR_ASSIGN',
        '^=': 'BXOR_ASSIGN',
        '~=': 'BNOT_ASSIGN',
        '>>': 'RSHIFT',
        '<<': 'LSHIFT',
        '&&': 'AND',
        '||': 'OR',
        '++': 'INC',
        '--': 'DEC',
    }
    SHORT3 = {
        '<<=': 'LSHIFT_ASSIGN',
        '>>=': 'RSHIFT_ASSIGN'
    }
    KEYWORDS = {'auto', 'break', 'case', 'char',
                'const', 'continue', 'default', 'do',
                'double', 'else', 'enum', 'extern',
                'float', 'for', 'goto', 'if',
                'int', 'long', 'register', 'return',
                'short', 'signed', 'sizeof', 'static',
                'struct', 'switch', 'typedef', 'union',
                'unsigned', 'void', 'volatile', 'while',
                'inline'}

    def __init__(self):
        self._tokens = []

    def lex_file(self, reader):
        contents = reader.read()
        source = Source(contents)
        source = self.remove_comments(source)
        self.lex(source)

    def remove_comments(self, source):
        NO_COMMENT = 0
        SINGLE_COMMENT = 1
        MULTI_COMMENT = 2
        IN_STRING = 3
        state = NO_COMMENT
        cleaned = ''
        while source.chars_left() > 0:
            if state == NO_COMMENT:
                match source.lookahead(2):
                    case '//':
                        state = SINGLE_COMMENT
                        source.advance(2)
                        continue
                    case '/*':
                        state = MULTI_COMMENT
                        source.advance(2)
                        continue
                match source.lookahead():
                    case '"':
                        state = IN_STRING
                        cleaned += source.advance()
                        continue
                cleaned += source.advance()
            elif state == SINGLE_COMMENT:
                match source.lookahead():
                    case '\n':
                        state = NO_COMMENT
                        cleaned += source.advance()
                        continue
                source.advance()
            elif state == MULTI_COMMENT:
                match source.lookahead(2):
                    case '*/':
                        state = NO_COMMENT
                        source.advance(2)
                        continue
                source.advance()
            elif state == IN_STRING:
                match source.lookahead(2):
                    case '\\\\':
                        cleaned += source.advance(2)
                        continue
                    case '\\"':
                        cleaned += source.advance(2)
                        continue
                match source.lookahead():
                    case '"':
                        state = NO_COMMENT
                        cleaned += source.advance()
                        continue
                cleaned += source.advance()
        return Source(cleaned)

    def lex(self, source):
        while source.chars_left() > 0:
            ch = source.lookahead()
            if ch.lower() in '_abcdefghijklmnopqrtstuvwxyz':
                self.lex_ident(source)
            elif ch in '0123456789':
                self.lex_num(source)
            elif ch == '"':
                self.lex_string(source)
            elif ch == '\'':
                self.lex_char(source)
            elif source.lookahead(3) in Lexer.SHORT3:
                self.lex_short3(source)
            elif source.lookahead(2) in Lexer.SHORT2:
                self.lex_short2(source)
            elif ch in Lexer.SHORT1:
                self.lex_short1(source)
            elif ch == '#':
                self.lex_macro(source)
            else:
                source.advance()

    def lex_ident(self, source):
        IDENT_PART = set(list('_abcdefghijklmnopqrtstuvwxyz0123456789'))
        ident = ''
        while source.chars_left() > 0 and source.lookahead().lower() in IDENT_PART:
            ident += source.lookahead()
            source.advance()
        if ident in Lexer.KEYWORDS:
            type_ = f'KW_{ident.upper()}'
            self._tokens.append((ident, type_))
        else:
            self._tokens.append((ident, 'IDENT'))

    def lex_num(self, source):
        VALID = set(list('0123456789abcdefghijklmnopqrstuvwxyz.'))
        num = ''
        float_ = False
        hex_ = False
        while source.lookahead().lower() in VALID:
            if source.lookahead() == '.' or source.lookahead().lower() == 'e':
                float_ = True
            if source.lookahead().lower() == 'x':
                hex_ = True
            num += source.lookahead()
            source.advance()
        type_ = 'FLOAT' if float_ and not hex_ else 'INT'
        self._tokens.append((num, type_))

    def lex_string(self, source):
        source.advance()
        string = '"'
        while source.chars_left() > 0:
            match source.lookahead(2):
                case '\\\\':
                    string += source.advance(2)
                    continue
                case '\\"':
                    string += source.advance(2)
                    continue
            match source.lookahead():
                case '"':
                    string += source.advance()
                    break
            string += source.advance()
        self._tokens.append((string, 'STRING'))

    def lex_char(self, source):
        source.advance()
        char = '\''
        char += source.advance(2)
        self._tokens.append((char, 'CHAR'))

    def lex_short3(self, source):
        lexeme = source.advance(3)
        type_ = Lexer.SHORT3[lexeme]
        self._tokens.append((lexeme, type_))

    def lex_short2(self, source):
        lexeme = source.advance(2)
        type_ = Lexer.SHORT2[lexeme]
        self._tokens.append((lexeme, type_))

    def lex_short1(self, source):
        lexeme = source.advance()
        type_ = Lexer.SHORT1[lexeme]
        self._tokens.append((lexeme, type_))

    def lex_macro(self, source):
        while source.chars_left() > 0 and source.lookahead() != '\n':
            source.advance()



if __name__ == '__main__':
    with open('main.c', 'r') as f:
        lex = Lexer()
        lex.lex_file(f)
        print(lex._tokens)
