"""Tests for hash_ast_args — order-independent argument hashing."""

import ast
import unittest

from cylint.diff.ast_hash import hash_ast_args, hash_ast_subtree


class TestHashAstArgs(unittest.TestCase):

    def _parse_args(self, expr: str) -> list[ast.expr]:
        """Parse a function call and return its arguments."""
        tree = ast.parse(expr, mode="eval")
        call = tree.body
        return call.args

    def test_order_independent_string_args(self):
        """groupBy("a", "b") and groupBy("b", "a") should produce the same hash."""
        args1 = self._parse_args('f("a", "b")')
        args2 = self._parse_args('f("b", "a")')
        self.assertEqual(hash_ast_args(args1), hash_ast_args(args2))

    def test_different_args_different_hash(self):
        """groupBy("a", "b") and groupBy("a", "c") should differ."""
        args1 = self._parse_args('f("a", "b")')
        args2 = self._parse_args('f("a", "c")')
        self.assertNotEqual(hash_ast_args(args1), hash_ast_args(args2))

    def test_single_arg_consistency(self):
        """Single arg hashed via hash_ast_args should be deterministic."""
        args1 = self._parse_args('f("date")')
        args2 = self._parse_args('f("date")')
        self.assertEqual(hash_ast_args(args1), hash_ast_args(args2))

    def test_expression_args_order_independent(self):
        """Complex expressions should also be order-independent."""
        args1 = self._parse_args('f(a.id == b.id, a.date == b.date)')
        args2 = self._parse_args('f(a.date == b.date, a.id == b.id)')
        self.assertEqual(hash_ast_args(args1), hash_ast_args(args2))

    def test_empty_args(self):
        """Empty argument list should produce a consistent hash."""
        h1 = hash_ast_args([])
        h2 = hash_ast_args([])
        self.assertEqual(h1, h2)

    def test_additional_arg_changes_hash(self):
        """Adding an argument should change the hash."""
        args1 = self._parse_args('f("date")')
        args2 = self._parse_args('f("date", "status")')
        self.assertNotEqual(hash_ast_args(args1), hash_ast_args(args2))


if __name__ == "__main__":
    unittest.main()
