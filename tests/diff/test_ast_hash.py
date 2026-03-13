"""Tests for ast_hash.py — position-independent AST hashing."""

import ast
import unittest

from cylint.diff.ast_hash import hash_ast_subtree


class TestHashAstSubtree(unittest.TestCase):
    def test_same_expression_same_hash(self):
        a = ast.parse("x > 100", mode="eval").body
        b = ast.parse("x > 100", mode="eval").body
        self.assertEqual(hash_ast_subtree(a), hash_ast_subtree(b))

    def test_different_expression_different_hash(self):
        a = ast.parse("x > 100", mode="eval").body
        b = ast.parse("x > 500", mode="eval").body
        self.assertNotEqual(hash_ast_subtree(a), hash_ast_subtree(b))

    def test_position_independent(self):
        """Same expression at different source locations should hash the same."""
        src1 = "x = foo(1)\ny = bar(x > 100)"
        src2 = "a = 1\nb = 2\nc = 3\nz = bar(x > 100)"
        tree1 = ast.parse(src1)
        tree2 = ast.parse(src2)
        # Extract the `x > 100` from each
        call1 = tree1.body[1].value  # bar(x > 100)
        call2 = tree2.body[3].value
        arg1 = call1.args[0]
        arg2 = call2.args[0]
        self.assertEqual(hash_ast_subtree(arg1), hash_ast_subtree(arg2))

    def test_hash_is_16_char_hex(self):
        node = ast.parse("x + 1", mode="eval").body
        h = hash_ast_subtree(node)
        self.assertEqual(len(h), 16)
        int(h, 16)  # should not raise

    def test_whitespace_irrelevant(self):
        a = ast.parse("x>100", mode="eval").body
        b = ast.parse("x >  100", mode="eval").body
        self.assertEqual(hash_ast_subtree(a), hash_ast_subtree(b))


if __name__ == "__main__":
    unittest.main()
