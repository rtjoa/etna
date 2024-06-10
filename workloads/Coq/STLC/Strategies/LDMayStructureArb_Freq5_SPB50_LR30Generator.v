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
      | (0, 3, 3) => 90
      | (0, 3, 5) => 90
      | (0, 3, 6) => 91
      | (0, 5, 3) => 89
      | (0, 5, 5) => 88
      | (0, 5, 6) => 88
      | (0, 6, 3) => 90
      | (0, 6, 5) => 80
      | (0, 6, 6) => 28
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 77
      | (0, 3, 6) => 58
      | (0, 5, 3) => 0
      | (0, 5, 5) => 2
      | (0, 5, 6) => 6
      | (0, 6, 3) => 0
      | (0, 6, 5) => 88
      | (0, 6, 6) => 92
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 16
      | (1, 3, 5) => 3
      | (1, 3, 6) => 79
      | (1, 5, 3) => 4
      | (1, 5, 5) => 65
      | (1, 5, 6) => 21
      | (1, 6, 3) => 51
      | (1, 6, 5) => 34
      | (1, 6, 6) => 58
      | (2, 3, 3) => 5
      | (2, 3, 5) => 5
      | (2, 3, 6) => 0
      | (2, 5, 3) => 56
      | (2, 5, 5) => 21
      | (2, 5, 6) => 55
      | (2, 6, 3) => 2
      | (2, 6, 5) => 8
      | (2, 6, 6) => 11
      | (3, 3, 3) => 20
      | (3, 3, 5) => 35
      | (3, 3, 6) => 22
      | (3, 5, 3) => 1
      | (3, 5, 5) => 17
      | (3, 5, 6) => 0
      | (3, 6, 3) => 42
      | (3, 6, 5) => 55
      | (3, 6, 6) => 42
      | (4, 0, 3) => 4
      | (4, 0, 5) => 0
      | (4, 0, 6) => 5
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 5
      | (1, 3, 6) => 16
      | (1, 5, 3) => 0
      | (1, 5, 5) => 9
      | (1, 5, 6) => 5
      | (1, 6, 3) => 0
      | (1, 6, 5) => 2
      | (1, 6, 6) => 2
      | (2, 3, 3) => 0
      | (2, 3, 5) => 6
      | (2, 3, 6) => 1
      | (2, 5, 3) => 8
      | (2, 5, 5) => 50
      | (2, 5, 6) => 60
      | (2, 6, 3) => 0
      | (2, 6, 5) => 9
      | (2, 6, 6) => 0
      | (3, 3, 3) => 20
      | (3, 3, 5) => 44
      | (3, 3, 6) => 62
      | (3, 5, 3) => 0
      | (3, 5, 5) => 24
      | (3, 5, 6) => 0
      | (3, 6, 3) => 42
      | (3, 6, 5) => 65
      | (3, 6, 6) => 59
      | (4, 0, 3) => 4
      | (4, 0, 5) => 0
      | (4, 0, 6) => 72
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 48
      | (1, 3, 5) => 15
      | (1, 3, 6) => 92
      | (1, 5, 3) => 90
      | (1, 5, 5) => 83
      | (1, 5, 6) => 38
      | (1, 6, 3) => 90
      | (1, 6, 5) => 82
      | (1, 6, 6) => 92
      | (2, 3, 3) => 12
      | (2, 3, 5) => 11
      | (2, 3, 6) => 0
      | (2, 5, 3) => 77
      | (2, 5, 5) => 62
      | (2, 5, 6) => 62
      | (2, 6, 3) => 1
      | (2, 6, 5) => 75
      | (2, 6, 6) => 39
      | (3, 3, 3) => 84
      | (3, 3, 5) => 60
      | (3, 3, 6) => 70
      | (3, 5, 3) => 1
      | (3, 5, 5) => 57
      | (3, 5, 6) => 0
      | (3, 6, 3) => 57
      | (3, 6, 5) => 41
      | (3, 6, 6) => 48
      | (4, 0, 3) => 89
      | (4, 0, 5) => 0
      | (4, 0, 6) => 82
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 97
      | (1, 3, 5) => 94
      | (1, 3, 6) => 88
      | (1, 5, 3) => 63
      | (1, 5, 5) => 16
      | (1, 5, 6) => 88
      | (1, 6, 3) => 95
      | (1, 6, 5) => 87
      | (1, 6, 6) => 94
      | (2, 3, 3) => 95
      | (2, 3, 5) => 91
      | (2, 3, 6) => 96
      | (2, 5, 3) => 43
      | (2, 5, 5) => 61
      | (2, 5, 6) => 19
      | (2, 6, 3) => 96
      | (2, 6, 5) => 83
      | (2, 6, 6) => 93
      | (3, 3, 3) => 25
      | (3, 3, 5) => 59
      | (3, 3, 6) => 33
      | (3, 5, 3) => 95
      | (3, 5, 5) => 78
      | (3, 5, 6) => 96
      | (3, 6, 3) => 57
      | (3, 6, 5) => 36
      | (3, 6, 6) => 50
      | (4, 0, 3) => 31
      | (4, 0, 5) => 96
      | (4, 0, 6) => 7
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
          
