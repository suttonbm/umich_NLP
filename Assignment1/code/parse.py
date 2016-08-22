from providedcode.transitionparser import TransitionParser
import sys

def main():
    if len(sys.argv) < 4:
        print """
        Usage:

        python parse.py in.model > out.conll

        Input can be provided manually via the command prompt or piped directly
        to the script using cat.
        """
    # END if

    if sys.stdin.isatty():
        rawtext = [raw_input("Please type a sentence!")]
    else:
        rawtext = sys.stdin.read()
    # END if

    out_filename = sys.argv[3]
    model_filename = sys.argv[1]

    try:
        tp = TransitionParser.load(model_filename)
        parsed = tp.parse(rawtext)

        with open(out_filename, 'w') as f:
            for p in parsed:
                f.write(p.to_conll(10).encode('utf-8'))
                f.write('\n')
            # END for
        # END with
    except Exception:
        "Error."
# END main

if __name__ == '__main__':
    main()