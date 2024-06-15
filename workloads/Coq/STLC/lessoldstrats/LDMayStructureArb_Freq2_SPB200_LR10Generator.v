From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 89
      | (0, 3, 5) => 52
      | (0, 3, 6) => 49
      | (0, 5, 3) => 87
      | (0, 5, 5) => 56
      | (0, 5, 6) => 54
      | (0, 6, 3) => 88
      | (0, 6, 5) => 42
      | (0, 6, 6) => 56
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 58
      | (0, 3, 6) => 60
      | (0, 5, 3) => 0
      | (0, 5, 5) => 47
      | (0, 5, 6) => 48
      | (0, 6, 3) => 0
      | (0, 6, 5) => 64
      | (0, 6, 6) => 53
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 40
      | (1, 3, 5) => 31
      | (1, 3, 6) => 33
      | (1, 5, 3) => 62
      | (1, 5, 5) => 42
      | (1, 5, 6) => 40
      | (1, 6, 3) => 54
      | (1, 6, 5) => 35
      | (1, 6, 6) => 37
      | (2, 3, 3) => 5
      | (2, 3, 5) => 23
      | (2, 3, 6) => 4
      | (2, 5, 3) => 53
      | (2, 5, 5) => 44
      | (2, 5, 6) => 51
      | (2, 6, 3) => 2
      | (2, 6, 5) => 20
      | (2, 6, 6) => 3
      | (3, 3, 3) => 46
      | (3, 3, 5) => 49
      | (3, 3, 6) => 46
      | (3, 5, 3) => 0
      | (3, 5, 5) => 22
      | (3, 5, 6) => 0
      | (3, 6, 3) => 48
      | (3, 6, 5) => 50
      | (3, 6, 6) => 48
      | (4, 0, 3) => 24
      | (4, 0, 5) => 0
      | (4, 0, 6) => 25
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 24
      | (1, 3, 6) => 42
      | (1, 5, 3) => 2
      | (1, 5, 5) => 47
      | (1, 5, 6) => 50
      | (1, 6, 3) => 0
      | (1, 6, 5) => 27
      | (1, 6, 6) => 39
      | (2, 3, 3) => 0
      | (2, 3, 5) => 24
      | (2, 3, 6) => 3
      | (2, 5, 3) => 12
      | (2, 5, 5) => 45
      | (2, 5, 6) => 42
      | (2, 6, 3) => 0
      | (2, 6, 5) => 21
      | (2, 6, 6) => 2
      | (3, 3, 3) => 46
      | (3, 3, 5) => 49
      | (3, 3, 6) => 53
      | (3, 5, 3) => 0
      | (3, 5, 5) => 22
      | (3, 5, 6) => 0
      | (3, 6, 3) => 48
      | (3, 6, 5) => 48
      | (3, 6, 6) => 51
      | (4, 0, 3) => 24
      | (4, 0, 5) => 0
      | (4, 0, 6) => 63
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 83
      | (1, 3, 5) => 73
      | (1, 3, 6) => 72
      | (1, 5, 3) => 74
      | (1, 5, 5) => 64
      | (1, 5, 6) => 64
      | (1, 6, 3) => 80
      | (1, 6, 5) => 71
      | (1, 6, 6) => 59
      | (2, 3, 3) => 24
      | (2, 3, 5) => 58
      | (2, 3, 6) => 26
      | (2, 5, 3) => 62
      | (2, 5, 5) => 56
      | (2, 5, 6) => 51
      | (2, 6, 3) => 20
      | (2, 6, 5) => 62
      | (2, 6, 6) => 19
      | (3, 3, 3) => 58
      | (3, 3, 5) => 52
      | (3, 3, 6) => 53
      | (3, 5, 3) => 3
      | (3, 5, 5) => 54
      | (3, 5, 6) => 2
      | (3, 6, 3) => 54
      | (3, 6, 5) => 51
      | (3, 6, 6) => 52
      | (4, 0, 3) => 81
      | (4, 0, 5) => 0
      | (4, 0, 6) => 68
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 81
      | (1, 3, 5) => 61
      | (1, 3, 6) => 59
      | (1, 5, 3) => 57
      | (1, 5, 5) => 45
      | (1, 5, 6) => 43
      | (1, 6, 3) => 81
      | (1, 6, 5) => 59
      | (1, 6, 6) => 72
      | (2, 3, 3) => 93
      | (2, 3, 5) => 75
      | (2, 3, 6) => 90
      | (2, 5, 3) => 65
      | (2, 5, 5) => 54
      | (2, 5, 6) => 56
      | (2, 6, 3) => 93
      | (2, 6, 5) => 75
      | (2, 6, 6) => 91
      | (3, 3, 3) => 49
      | (3, 3, 5) => 50
      | (3, 3, 6) => 47
      | (3, 5, 3) => 95
      | (3, 5, 5) => 77
      | (3, 5, 6) => 94
      | (3, 6, 3) => 49
      | (3, 6, 5) => 51
      | (3, 6, 6) => 49
      | (4, 0, 3) => 36
      | (4, 0, 5) => 96
      | (4, 0, 6) => 31
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
