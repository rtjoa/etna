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
      | (0, 3, 5) => 58
      | (0, 3, 6) => 59
      | (0, 5, 3) => 87
      | (0, 5, 5) => 40
      | (0, 5, 6) => 52
      | (0, 6, 3) => 88
      | (0, 6, 5) => 58
      | (0, 6, 6) => 53
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 48
      | (0, 3, 6) => 46
      | (0, 5, 3) => 0
      | (0, 5, 5) => 60
      | (0, 5, 6) => 51
      | (0, 6, 3) => 0
      | (0, 6, 5) => 48
      | (0, 6, 6) => 52
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 60
      | (1, 3, 5) => 18
      | (1, 3, 6) => 41
      | (1, 5, 3) => 69
      | (1, 5, 5) => 42
      | (1, 5, 6) => 46
      | (1, 6, 3) => 28
      | (1, 6, 5) => 46
      | (1, 6, 6) => 39
      | (2, 3, 3) => 2
      | (2, 3, 5) => 24
      | (2, 3, 6) => 2
      | (2, 5, 3) => 53
      | (2, 5, 5) => 45
      | (2, 5, 6) => 43
      | (2, 6, 3) => 2
      | (2, 6, 5) => 25
      | (2, 6, 6) => 1
      | (3, 3, 3) => 47
      | (3, 3, 5) => 48
      | (3, 3, 6) => 47
      | (3, 5, 3) => 0
      | (3, 5, 5) => 20
      | (3, 5, 6) => 0
      | (3, 6, 3) => 48
      | (3, 6, 5) => 52
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
      | (1, 3, 5) => 21
      | (1, 3, 6) => 18
      | (1, 5, 3) => 2
      | (1, 5, 5) => 41
      | (1, 5, 6) => 47
      | (1, 6, 3) => 0
      | (1, 6, 5) => 40
      | (1, 6, 6) => 36
      | (2, 3, 3) => 0
      | (2, 3, 5) => 11
      | (2, 3, 6) => 1
      | (2, 5, 3) => 10
      | (2, 5, 5) => 48
      | (2, 5, 6) => 48
      | (2, 6, 3) => 0
      | (2, 6, 5) => 25
      | (2, 6, 6) => 1
      | (3, 3, 3) => 47
      | (3, 3, 5) => 49
      | (3, 3, 6) => 52
      | (3, 5, 3) => 0
      | (3, 5, 5) => 19
      | (3, 5, 6) => 1
      | (3, 6, 3) => 48
      | (3, 6, 5) => 45
      | (3, 6, 6) => 52
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
      | (1, 3, 3) => 87
      | (1, 3, 5) => 71
      | (1, 3, 6) => 77
      | (1, 5, 3) => 70
      | (1, 5, 5) => 67
      | (1, 5, 6) => 66
      | (1, 6, 3) => 86
      | (1, 6, 5) => 60
      | (1, 6, 6) => 61
      | (2, 3, 3) => 19
      | (2, 3, 5) => 48
      | (2, 3, 6) => 14
      | (2, 5, 3) => 67
      | (2, 5, 5) => 47
      | (2, 5, 6) => 54
      | (2, 6, 3) => 26
      | (2, 6, 5) => 38
      | (2, 6, 6) => 18
      | (3, 3, 3) => 56
      | (3, 3, 5) => 54
      | (3, 3, 6) => 53
      | (3, 5, 3) => 1
      | (3, 5, 5) => 57
      | (3, 5, 6) => 1
      | (3, 6, 3) => 55
      | (3, 6, 5) => 51
      | (3, 6, 6) => 51
      | (4, 0, 3) => 81
      | (4, 0, 5) => 0
      | (4, 0, 6) => 67
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 57
      | (1, 3, 5) => 72
      | (1, 3, 6) => 58
      | (1, 5, 3) => 59
      | (1, 5, 5) => 43
      | (1, 5, 6) => 36
      | (1, 6, 3) => 78
      | (1, 6, 5) => 55
      | (1, 6, 6) => 68
      | (2, 3, 3) => 93
      | (2, 3, 5) => 82
      | (2, 3, 6) => 92
      | (2, 5, 3) => 61
      | (2, 5, 5) => 58
      | (2, 5, 6) => 54
      | (2, 6, 3) => 93
      | (2, 6, 5) => 80
      | (2, 6, 6) => 92
      | (3, 3, 3) => 49
      | (3, 3, 5) => 50
      | (3, 3, 6) => 48
      | (3, 5, 3) => 95
      | (3, 5, 5) => 78
      | (3, 5, 6) => 94
      | (3, 6, 3) => 49
      | (3, 6, 5) => 52
      | (3, 6, 6) => 48
      | (4, 0, 3) => 34
      | (4, 0, 5) => 96
      | (4, 0, 6) => 33
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
          
