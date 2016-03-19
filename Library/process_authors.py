import re

whitespace_re = re.compile(r'[ \t]+')
brackets_re = re.compile(r'\(([^()])*\)')  # ( .. ) with no embedded ( or )
leftbracket_re = re.compile(r'\([^)]*')  # left bracket that is never closed

def process_authors(authors):
    while True:
        # Remove all possible balanced parentheses
        brackets_removed = brackets_re.sub('', authors)
        if brackets_removed != authors:
            authors = brackets_removed
            continue
        # Remove all unbalanced parentheses
        leftbracket_removed = leftbracket_re.sub('', authors)
        if leftbracket_removed == authors:
            break
        authors = leftbracket_removed
    authors = whitespace_re.sub(' ', authors)
    return authors.lower()

def split_authors(authors):
    return map(str.strip, authors.lower().split(','))

if __name__ == '__main__':
    raw = 'Ron Donagi (UPenn), Burt   Ovrut (UPenn), Tony Pantev (UPenn), Dan ('
    processed = process_authors(raw)
    split = split_authors(processed)
    print 'raw:', raw
    print 'processed:', processed
    print 'split:', split

